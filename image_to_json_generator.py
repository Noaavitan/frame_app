import os
import json
import exifread
import subprocess
import re
from xml.etree import ElementTree as ET
import math
from datetime import datetime
from pathlib import Path
from shutil import which
import shutil
import base64


def _read_drone_type_from_config(folder_path: str) -> str:
    """
    קורא את config.json (אם קיים) ומחזיר את שדה 'drone_type'.
    אם לא קיים או ריק – מחזיר מחרוזת ברירת מחדל.
    """
    cfg_path = os.path.join(folder_path, "config.json")
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                dt = (cfg.get("drone_type") or "").strip()
                if dt:
                    return dt
    except Exception as e:
        print(f"Warning: could not read config.json: {e}")
    return "Unknown platform"


# -------------------------------
# ExifTool resolution & wrapper
# -------------------------------

def resolve_exiftool_path() -> str | None:
    """
    מחפש את exiftool.exe לפי הסדר:
    1) משתנה סביבה EXIFTOOL_PATH
    2) ב-PATH (which)
    3) נתיבים יחסיים נפוצים ליד הקובץ הזה
    """
    # 1) EXIFTOOL_PATH מהסביבה
    env_p = os.environ.get("EXIFTOOL_PATH")
    if env_p and os.path.exists(env_p):
        return env_p

    # 2) PATH
    w = which("exiftool")
    if w and os.path.exists(w):
        return w

    # 3) חיפוש מקומי ליד הקובץ
    here = Path(__file__).resolve().parent
    local_candidates = [
        here / "exiftool-13.30_64" / "exiftool.exe",
        here / "exiftool-13.32_64" / "exiftool.exe",
        here / "exiftool.exe",
    ]
    for c in local_candidates:
        if c.is_file():
            return str(c)

    return None

EXIFTOOL_PATH = resolve_exiftool_path()

def run_exiftool(args: list[str]) -> subprocess.CompletedProcess:
    """
    הרצה בטוחה של ExifTool (ללא shell=True).
    זורק חריגות אם אין EXIFTOOL או אם ההרצה נכשלה (check=True).
    """
    if not EXIFTOOL_PATH:
        raise FileNotFoundError(
            "ExifTool לא נמצא. עדכן את EXIFTOOL_PATH, הוסף ל-PATH, או ודא שהקובץ exiftool.exe קיים ליד הסקריפט."
        )
    cmd = [EXIFTOOL_PATH] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


# -------------------------------
# EXIF / XMP helpers
# -------------------------------

def get_decimal_from_dms(dms, ref):
    """
    Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees
    dms: tuple of (degrees, minutes, seconds)
    ref: reference direction ('N', 'S', 'E', 'W')
    """
    try:
        degrees = float(dms[0].num) / float(dms[0].den)
        minutes = float(dms[1].num) / float(dms[1].den)
        seconds = float(dms[2].num) / float(dms[2].den)
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 6)
    except Exception as e:
        print(f"Error converting DMS to decimal: {e}")
        return None

def extract_gps_info_from_tags(tags):
    """
    Extract GPS coordinates from EXIF tags
    Returns coordinates in WGS84 Decimal Degrees format
    """
    try:
        if not all(key in tags for key in ['GPS GPSLatitude', 'GPS GPSLatitudeRef',
                                           'GPS GPSLongitude', 'GPS GPSLongitudeRef']):
            print("Missing required GPS tags")
            return None, None

        lat = tags['GPS GPSLatitude']
        lat_ref = tags['GPS GPSLatitudeRef'].printable
        lat_decimal = get_decimal_from_dms(lat.values, lat_ref)

        lon = tags['GPS GPSLongitude']
        lon_ref = tags['GPS GPSLongitudeRef'].printable
        lon_decimal = get_decimal_from_dms(lon.values, lon_ref)

        if lat_decimal is None or lon_decimal is None:
            print("Failed to convert GPS coordinates")
            return None, None

        if not (-90 <= lat_decimal <= 90) or not (-180 <= lon_decimal <= 180):
            print(f"Invalid coordinate values: lat={lat_decimal}, lon={lon_decimal}")
            return None, None

        print(f"Extracted GPS coordinates: {lat_decimal}, {lon_decimal}")
        return lat_decimal, lon_decimal

    except Exception as e:
        print(f"Error extracting GPS info: {e}")
        return None, None

def get_float(tag_name, tags, default=0.0):
    val = tags.get(tag_name)
    if val:
        try:
            return float(str(val))
        except:
            pass
    return default

def extract_xmp_metadata(image_path):
    """Extract XMP metadata from image file"""
    try:
        with open(image_path, "rb") as f:
            jpeg_data = f.read()
            xmp_match = re.search(br"<x:xmpmeta[^>]*>.*?</x:xmpmeta>", jpeg_data, re.DOTALL)
            if not xmp_match:
                return None

            xmp_data = xmp_match.group(0).decode("utf-8", errors="ignore")
            xmp_root = ET.fromstring(xmp_data)
            ns = {
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "drone-dji": "http://www.dji.com/drone-dji/1.0/"
            }
            return xmp_root, ns
    except Exception as e:
        print(f"Error extracting XMP metadata: {str(e)}")
        return None

def calculate_resolution(width, height, fov_x, fov_y, altitude):
    """
    Calculate resolution in meters per pixel
    """
    fov_x_rad = math.radians(fov_x)
    fov_y_rad = math.radians(fov_y)
    ground_width = 2 * altitude * math.tan(fov_x_rad / 2)
    ground_height = 2 * altitude * math.tan(fov_y_rad / 2)
    resolution_x = ground_width / width if width else 0
    resolution_y = ground_height / height if height else 0
    return round((resolution_x + resolution_y) / 2, 5) if (resolution_x and resolution_y) else 0.0

def get_los_fields(image_path):
    """
    מחלץ זוויות/כיוונים של הגימבל באמצעות ExifTool.
    מחזיר always dict עם מפתחות losAzimuth/losPitch/losRoll.
    """
    try:
        # -n לקבלת ערכים נומריים (ללא '+'), -json להחזרה במבנה JSON
        cp = run_exiftool(["-n", "-json", "-GimbalYawDegree", "-GimbalPitchDegree", "-GimbalRollDegree", image_path])
        data = json.loads(cp.stdout)[0] if cp.stdout else {}

        def to_float(val):
            try:
                return float(val)
            except Exception:
                return 0.0

        return {
            "losAzimuth": to_float(data.get("GimbalYawDegree")),
            "losPitch": to_float(data.get("GimbalPitchDegree")),
            "losRoll": to_float(data.get("GimbalRollDegree")),
        }
    except FileNotFoundError as e:
        print(f"Warning: ExifTool not found: {e}")
        return {"losAzimuth": 0.0, "losPitch": 0.0, "losRoll": 0.0}
    except subprocess.CalledProcessError as e:
        print(f"Warning: ExifTool failed: {e.stderr.strip() if e.stderr else e}")
        return {"losAzimuth": 0.0, "losPitch": 0.0, "losRoll": 0.0}
    except Exception as e:
        print(f"Warning: Could not extract LOS fields: {e}")
        return {"losAzimuth": 0.0, "losPitch": 0.0, "losRoll": 0.0}

def extract_relative_altitude(image_path):
    xmp_data = extract_xmp_metadata(image_path)
    if xmp_data is None:
        return 0.0
    xmp_root, ns = xmp_data
    desc = xmp_root.find(".//rdf:Description", ns)
    if desc is None:
        return 0.0
    val = desc.attrib.get(f"{{{ns['drone-dji']}}}RelativeAltitude")
    if val is None:
        return 0.0
    try:
        return float(val.lstrip("+"))
    except ValueError:
        return 0.0


# -------------------------------
# JSON builders
# -------------------------------

def build_basic_data(filename, tags, full_path):
    width = int(str(tags.get("EXIF ExifImageWidth", "0")))
    height = int(str(tags.get("EXIF ExifImageLength", "0")))

    try:
        imaging_time = str(tags.get("EXIF DateTimeOriginal", ""))
        if imaging_time:
            dt = datetime.strptime(imaging_time, "%Y:%m:%d %H:%M:%S")
            imaging_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Warning: Could not format imaging time: {e}")
        imaging_time = ""

    try:
        print(f"Attempting to calculate resolution for: {filename}")
        print(f"Image dimensions: {width}x{height}")

        xmp_data = extract_xmp_metadata(full_path)
        if xmp_data is not None:
            print("Found XMP metadata section")
            xmp_root, ns = xmp_data
            desc = xmp_root.find(".//rdf:Description", ns)
            if desc is not None:
                print("Found XMP Description")
                val = desc.attrib.get(f"{{{ns['drone-dji']}}}RelativeAltitude")
                if val is not None:
                    print(f"Found RelativeAltitude: {val}")
                    altitude = float(val.lstrip("+"))
                    print(f"Using altitude: {altitude} meters")
                    resolution = calculate_resolution(width, height, 82.9, 52.5, altitude)
                    print(f"Calculated resolution: {resolution} meters/pixel")
                else:
                    print("RelativeAltitude not found in XMP")
                    resolution = 0.0
            else:
                print("XMP Description not found")
                resolution = 0.0
        else:
            print("XMP metadata section not found")
            resolution = 0.0
    except Exception as e:
        print(f"Error calculating resolution: {str(e)}")
        resolution = 0.0

    return {
        "id": os.path.splitext(filename)[0],
        "sensorName": "Modash",
        "sensorType": "VIS",
        "imageFile": filename,
        "imagingTime": imaging_time,
        "prevImagingTime": None,
        "nextImagingTime": None,
        "height": height,
        "width": width,
        "resolution": resolution
    }

def build_camera_data(tags):
    width = int(str(tags.get("EXIF ExifImageWidth", "0")))
    height = int(str(tags.get("EXIF ExifImageLength", "0")))
    try:
        focal_35mm = float(str(tags.get("EXIF FocalLengthIn35mmFilm", "0")))
        fx = (focal_35mm / 36.0) * width
        fy = (focal_35mm / 24.0) * height
    except:
        fx, fy = 0.0, 0.0
    return {
        "focalLengthInPixelsX": fx,
        "focalLengthInPixelsY": fy,
        "foVX": 82.9,
        "foVY": 52.5,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "alpha": 0.0,
        "cameraMake": str(tags.get("Image Make", "")),
        "cameraModel": str(tags.get("Image Model", "")),
        "focalId": None,
        "exposureDuration": None,
        "fnumber": None
    }

def build_camera_position(tags, lat, lon, image_path):
    def get_altitude(tag_name, default=0.0):
        val = tags.get(tag_name)
        try:
            if val and hasattr(val, 'values'):
                ratio = val.values[0]
                return ratio.num / ratio.den
        except Exception:
            pass
        return default

    los_fields = get_los_fields(image_path)
    relative_alt = extract_relative_altitude(image_path)

    return {
        "gpsLatitude": lat,
        "gpsLongitude": lon,
        "gpsAltitude": get_altitude("GPS GPSAltitude", 0.0),
        "relativeAltitude": relative_alt,
        "losAzimuth": los_fields["losAzimuth"],
        "losPitch": los_fields["losPitch"],
        "losRoll": los_fields["losRoll"]
    }

def build_platform_data(tags, drone_type: str):
    return {
        "platformName": drone_type,
        "platformId": None,
        "trueCourse": get_float("GPS GPSTrack", tags, 0.0),
        "groundSpeed": 0.01,
        "mslAltitude": get_float("GPS GPSAltitude", tags, 0.0),
        "platformYaw": 0.0,
        "platformPitch": 0.0,
        "platformRoll": 0.0
    }

def build_operational_data():
    return {
        "missionNumber": None,
        "operationUnit": "Padam"
    }

def build_sensor_specific_data():
    return {
        "state": "0",
        "sixDofSource": None,
        "groundRef": None
    }

def build_json_structure(filename, tags, lat, lon, full_path, drone_type: str):
    return {
        "BasicData": build_basic_data(filename, tags, full_path),
        "CameraData": build_camera_data(tags),
        "CameraPosition": build_camera_position(tags, lat, lon, full_path),
        "PlatformData": build_platform_data(tags, drone_type),
        "Operational": build_operational_data(),
        "SensorSpecificData": build_sensor_specific_data(),
    }


# -------------------------------
# JPW / QGIS helpers
# -------------------------------

def create_jpw_from_json(json_filepath):
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_filepath} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File {json_filepath} is not a valid JSON.")
        return

    try:
        image_file = data['BasicData']['imageFile']
        width_pixels = data['BasicData']['width']
        height_pixels = data['BasicData']['height']
        resolution_meters_per_pixel = data['BasicData']['resolution']
        center_lat = data['CameraPosition']['gpsLatitude']
        center_lon = data['CameraPosition']['gpsLongitude']
    except KeyError as e:
        print(f"Error: JSON structure is invalid. Missing key: {e}")
        return

    lat_radians = math.radians(center_lat)
    meters_per_deg_lon = 111320 * math.cos(lat_radians)
    A = resolution_meters_per_pixel / meters_per_deg_lon

    meters_per_deg_lat = 111320
    E = -resolution_meters_per_pixel / meters_per_deg_lat

    B = 0.0
    D = 0.0

    offset_x_deg = (width_pixels / 2) * A
    offset_y_deg = (height_pixels / 2) * abs(E)
    C = center_lon - offset_x_deg
    F = center_lat + offset_y_deg

    jpw_filename = os.path.splitext(image_file)[0] + '.jpw'
    jpw_filepath = os.path.join(os.path.dirname(json_filepath), jpw_filename)

    jpw_content = f"{A}\n{D}\n{B}\n{E}\n{C}\n{F}"

    try:
        with open(jpw_filepath, 'w', encoding='utf-8') as f:
            f.write(jpw_content)
        print(f"JPW file created successfully: {jpw_filepath}")
    except IOError as e:
        print(f"Error writing JPW file: {e}")

def prepare_data_for_qgis(folder_path):
    """
    Creates TO_QGIS folder containing .jpg + .json + .jpw for each valid image.
    """
    output_dir = os.path.join(folder_path, "output")
    to_qgis_dir = os.path.join(folder_path, "TO_QGIS")
    os.makedirs(to_qgis_dir, exist_ok=True)

    for file in os.listdir(output_dir):
        if file.lower().endswith('.json'):
            json_path = os.path.join(output_dir, file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    image_file_name = data['BasicData']['imageFile']
            except Exception as e:
                print(f" Failed reading JSON {file}: {e}")
                continue

            image_path = os.path.join(folder_path, image_file_name)

            if not os.path.exists(image_path):
                print(f" Image not found for JSON: {image_file_name}")
                continue

            create_jpw_from_json(json_path)

            jpw_file_name = os.path.splitext(image_file_name)[0] + ".jpw"
            jpw_path = os.path.join(output_dir, jpw_file_name)

            if not os.path.exists(jpw_path):
                print(f"️ JPW file not created for: {image_file_name}")
                continue

            try:
                shutil.copy2(image_path, os.path.join(to_qgis_dir, image_file_name))
                shutil.copy2(json_path, os.path.join(to_qgis_dir, file))
                shutil.copy2(jpw_path, os.path.join(to_qgis_dir, jpw_file_name))
                print(f" Copied {image_file_name}, {file}, and {jpw_file_name} to TO_QGIS")
            except Exception as e:
                print(f" Failed copying files to TO_QGIS: {e}")


# -------------------------------
# Main processing
# -------------------------------

def process_images_to_individual_json(folder_path):
    output_dir = os.path.join(folder_path, "output")
    fail_output_dir = os.path.join(folder_path, "fail_output")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fail_output_dir, exist_ok=True)

    # NEW: נקרא את שם הרחפן מהקונפיג שתיקיית העבודה קיבלה מה-pipeline
    drone_type = _read_drone_type_from_config(folder_path)
    print(f"Using platformName (drone_type): {drone_type}")

    print(f"Looking for images in: {folder_path}")
    files = os.listdir(folder_path)
    print(f"Found {len(files)} files")

    total_images = 0
    successful_extractions = 0
    failed_extractions = 0

    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg')):
            total_images += 1
            print(f"\nProcessing: {filename}")
            full_path = os.path.join(folder_path, filename)

            try:
                with open(full_path, 'rb') as img_file:
                    tags = exifread.process_file(img_file, details=True)
                    lat, lon = extract_gps_info_from_tags(tags)

                    los_fields = get_los_fields(full_path)
                    relative_alt = extract_relative_altitude(full_path)

                    has_los_fields = (los_fields["losAzimuth"] != 0.0 and
                                      los_fields["losPitch"] != 0.0)
                    has_relative_alt = relative_alt != 0.0

                    # UPDATED: מעביר drone_type
                    json_data = build_json_structure(filename, tags, lat, lon, full_path, drone_type)

                    if has_los_fields and has_relative_alt:
                        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                        successful_extractions += 1
                        print(f"✅ Successfully extracted critical fields: {output_path}")
                    else:
                        output_path = os.path.join(fail_output_dir, f"{os.path.splitext(filename)[0]}.json")
                        failed_extractions += 1
                        missing_fields = []
                        if not has_los_fields:
                            missing_fields.append("LOS fields (azimuth/pitch)")
                        if not has_relative_alt:
                            missing_fields.append("relative altitude")
                        print(f"⚠️ Missing critical fields ({', '.join(missing_fields)}): {output_path}")

                    with open(output_path, 'w', encoding='utf-8') as json_file:
                        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

            except Exception as e:
                failed_extractions += 1
                print(f"❌ Failed to process {filename}: {str(e)}")
                try:
                    with open(full_path, 'rb') as img_file:
                        tags = exifread.process_file(img_file, details=True)
                        lat, lon = extract_gps_info_from_tags(tags)
                        # UPDATED: גם במקרה שגיאה – עדיין מזריק platformName נכון
                        json_data = build_json_structure(filename, tags, lat, lon, full_path, drone_type)
                        output_path = os.path.join(fail_output_dir, f"{os.path.splitext(filename)[0]}.json")
                        with open(output_path, 'w', encoding='utf-8') as json_file:
                            json.dump(json_data, json_file, indent=4, ensure_ascii=False)
                except Exception as inner_e:
                    print(f"❌ Could not create JSON for {filename}: {str(inner_e)}")
        else:
            print(f"Skipping (not an image): {filename}")

    print("\nProcessing Statistics:")
    print(f"Total images processed: {total_images}")
    print(f"Successfully extracted all critical fields: {successful_extractions}")
    print(f"Failed or missing critical fields: {failed_extractions}")
    print(f"\nSuccessful extractions saved to: {output_dir}")
    print(f"Failed extractions saved to: {fail_output_dir}")


# -------------------------------
# Optional: LOG decoding (base64)
# -------------------------------

def extract_platform_data_from_log(log_file_path):
    """
    Extracts platform data from a base64-encoded DJI .LOG file.
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            base64_content = f.read()

        decoded_bytes = base64.b64decode(base64_content)
        decoded_text = decoded_bytes.decode('utf-8', errors='ignore')

        def extract_value(pattern, default=0.0):
            match = re.search(pattern, decoded_text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
            return default

        return {
            "platformName": "Padam DJI",
            "platformId": None,
            "trueCourse": extract_value(r'TrueCourse\s*[:=]\s*([0-9.]+)'),
            "groundSpeed": extract_value(r'GroundSpeed\s*[:=]\s*([0-9.]+)'),
            "mslAltitude": extract_value(r'AltitudeMSL\s*[:=]\s*([0-9.]+)'),
            "platformYaw": extract_value(r'Yaw\s*[:=]\s*([0-9.\-]+)'),
            "platformPitch": extract_value(r'Pitch\s*[:=]\s*([0-9.\-]+)'),
            "platformRoll": extract_value(r'Roll\s*[:=]\s*([0-9.\-]+)')
        }

    except Exception as e:
        print(f"❌ Failed to extract platform data: {e}")
        return {
            "platformName": "Padam DJI",
            "platformId": None,
            "trueCourse": 0.0,
            "groundSpeed": 0.0,
            "mslAltitude": 0.0,
            "platformYaw": 0.0,
            "platformPitch": 0.0,
            "platformRoll": 0.0
        }
