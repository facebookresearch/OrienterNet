"""Copied from opensfm.exif to minimize hard dependencies."""
from pathlib import Path
import json
import datetime
import logging
from codecs import encode, decode
from typing import Any, Dict, Optional, Tuple

import exifread

logger: logging.Logger = logging.getLogger(__name__)

inch_in_mm = 25.4
cm_in_mm = 10
um_in_mm = 0.001
default_projection = "perspective"
maximum_altitude = 1e4


def sensor_data():
    with (Path(__file__).parent / "sensor_data.json").open() as fid:
        data = json.load(fid)
    return {k.lower(): v for k, v in data.items()}


def eval_frac(value) -> Optional[float]:
    try:
        return float(value.num) / float(value.den)
    except ZeroDivisionError:
        return None


def gps_to_decimal(values, reference) -> Optional[float]:
    sign = 1 if reference in "NE" else -1
    degrees = eval_frac(values[0])
    minutes = eval_frac(values[1])
    seconds = eval_frac(values[2])
    if degrees is not None and minutes is not None and seconds is not None:
        return sign * (degrees + minutes / 60 + seconds / 3600)
    return None


def get_tag_as_float(tags, key, index: int = 0) -> Optional[float]:
    if key in tags:
        val = tags[key].values[index]
        if isinstance(val, exifread.utils.Ratio):
            ret_val = eval_frac(val)
            if ret_val is None:
                logger.error(
                    'The rational "{2}" of tag "{0:s}" at index {1:d} c'
                    "aused a division by zero error".format(key, index, val)
                )
            return ret_val
        else:
            return float(val)
    else:
        return None


def compute_focal(
    focal_35: Optional[float], focal: Optional[float], sensor_width, sensor_string
) -> Tuple[float, float]:
    if focal_35 is not None and focal_35 > 0:
        focal_ratio = focal_35 / 36.0  # 35mm film produces 36x24mm pictures.
    else:
        if not sensor_width:
            sensor_width = sensor_data().get(sensor_string, None)
        if sensor_width and focal:
            focal_ratio = focal / sensor_width
            focal_35 = 36.0 * focal_ratio
        else:
            focal_35 = 0.0
            focal_ratio = 0.0
    return focal_35, focal_ratio


def sensor_string(make: str, model: str) -> str:
    if make != "unknown":
        # remove duplicate 'make' information in 'model'
        model = model.replace(make, "")
    return (make.strip() + " " + model.strip()).strip().lower()


def unescape_string(s) -> str:
    return decode(encode(s, "latin-1", "backslashreplace"), "unicode-escape")


class EXIF:
    def __init__(
        self, fileobj, image_size_loader, use_exif_size=True, name=None
    ) -> None:
        self.image_size_loader = image_size_loader
        self.use_exif_size = use_exif_size
        self.fileobj = fileobj
        self.tags = exifread.process_file(fileobj, details=False)
        fileobj.seek(0)
        self.fileobj_name = self.fileobj.name if name is None else name

    def extract_image_size(self) -> Tuple[int, int]:
        if (
            self.use_exif_size
            and "EXIF ExifImageWidth" in self.tags
            and "EXIF ExifImageLength" in self.tags
        ):
            width, height = (
                int(self.tags["EXIF ExifImageWidth"].values[0]),
                int(self.tags["EXIF ExifImageLength"].values[0]),
            )
        elif (
            self.use_exif_size
            and "Image ImageWidth" in self.tags
            and "Image ImageLength" in self.tags
        ):
            width, height = (
                int(self.tags["Image ImageWidth"].values[0]),
                int(self.tags["Image ImageLength"].values[0]),
            )
        else:
            height, width = self.image_size_loader()
        return width, height

    def _decode_make_model(self, value) -> str:
        """Python 2/3 compatible decoding of make/model field."""
        if hasattr(value, "decode"):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return "unknown"
        else:
            return value

    def extract_make(self) -> str:
        # Camera make and model
        if "EXIF LensMake" in self.tags:
            make = self.tags["EXIF LensMake"].values
        elif "Image Make" in self.tags:
            make = self.tags["Image Make"].values
        else:
            make = "unknown"
        return self._decode_make_model(make)

    def extract_model(self) -> str:
        if "EXIF LensModel" in self.tags:
            model = self.tags["EXIF LensModel"].values
        elif "Image Model" in self.tags:
            model = self.tags["Image Model"].values
        else:
            model = "unknown"
        return self._decode_make_model(model)

    def extract_focal(self) -> Tuple[float, float]:
        make, model = self.extract_make(), self.extract_model()
        focal_35, focal_ratio = compute_focal(
            get_tag_as_float(self.tags, "EXIF FocalLengthIn35mmFilm"),
            get_tag_as_float(self.tags, "EXIF FocalLength"),
            self.extract_sensor_width(),
            sensor_string(make, model),
        )
        return focal_35, focal_ratio

    def extract_sensor_width(self) -> Optional[float]:
        """Compute sensor with from width and resolution."""
        if (
            "EXIF FocalPlaneResolutionUnit" not in self.tags
            or "EXIF FocalPlaneXResolution" not in self.tags
        ):
            return None
        resolution_unit = self.tags["EXIF FocalPlaneResolutionUnit"].values[0]
        mm_per_unit = self.get_mm_per_unit(resolution_unit)
        if not mm_per_unit:
            return None
        pixels_per_unit = get_tag_as_float(self.tags, "EXIF FocalPlaneXResolution")
        if pixels_per_unit is None:
            return None
        if pixels_per_unit <= 0.0:
            pixels_per_unit = get_tag_as_float(self.tags, "EXIF FocalPlaneYResolution")
            if pixels_per_unit is None or pixels_per_unit <= 0.0:
                return None
        units_per_pixel = 1 / pixels_per_unit
        width_in_pixels = self.extract_image_size()[0]
        return width_in_pixels * units_per_pixel * mm_per_unit

    def get_mm_per_unit(self, resolution_unit) -> Optional[float]:
        """Length of a resolution unit in millimeters.

        Uses the values from the EXIF specs in
        https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html

        Args:
            resolution_unit: the resolution unit value given in the EXIF
        """
        if resolution_unit == 2:  # inch
            return inch_in_mm
        elif resolution_unit == 3:  # cm
            return cm_in_mm
        elif resolution_unit == 4:  # mm
            return 1
        elif resolution_unit == 5:  # um
            return um_in_mm
        else:
            logger.warning(
                "Unknown EXIF resolution unit value: {}".format(resolution_unit)
            )
            return None

    def extract_orientation(self) -> int:
        orientation = 1
        if "Image Orientation" in self.tags:
            value = self.tags.get("Image Orientation").values[0]
            if type(value) == int and value != 0:
                orientation = value
        return orientation

    def extract_ref_lon_lat(self) -> Tuple[str, str]:
        if "GPS GPSLatitudeRef" in self.tags:
            reflat = self.tags["GPS GPSLatitudeRef"].values
        else:
            reflat = "N"
        if "GPS GPSLongitudeRef" in self.tags:
            reflon = self.tags["GPS GPSLongitudeRef"].values
        else:
            reflon = "E"
        return reflon, reflat

    def extract_lon_lat(self) -> Tuple[Optional[float], Optional[float]]:
        if "GPS GPSLatitude" in self.tags:
            reflon, reflat = self.extract_ref_lon_lat()
            lat = gps_to_decimal(self.tags["GPS GPSLatitude"].values, reflat)
            lon = gps_to_decimal(self.tags["GPS GPSLongitude"].values, reflon)
        else:
            lon, lat = None, None
        return lon, lat

    def extract_altitude(self) -> Optional[float]:
        if "GPS GPSAltitude" in self.tags:
            alt_value = self.tags["GPS GPSAltitude"].values[0]
            if isinstance(alt_value, exifread.utils.Ratio):
                altitude = eval_frac(alt_value)
            elif isinstance(alt_value, int):
                altitude = float(alt_value)
            else:
                altitude = None

            # Check if GPSAltitudeRef is equal to 1, which means GPSAltitude should be negative, reference: http://www.exif.org/Exif2-2.PDF#page=53
            if (
                "GPS GPSAltitudeRef" in self.tags
                and self.tags["GPS GPSAltitudeRef"].values[0] == 1
                and altitude is not None
            ):
                altitude = -altitude
        else:
            altitude = None
        return altitude

    def extract_dop(self) -> Optional[float]:
        if "GPS GPSDOP" in self.tags:
            return eval_frac(self.tags["GPS GPSDOP"].values[0])
        return None

    def extract_geo(self) -> Dict[str, Any]:
        altitude = self.extract_altitude()
        dop = self.extract_dop()
        lon, lat = self.extract_lon_lat()
        d = {}

        if lon is not None and lat is not None:
            d["latitude"] = lat
            d["longitude"] = lon
        if altitude is not None:
            d["altitude"] = min([maximum_altitude, altitude])
        if dop is not None:
            d["dop"] = dop
        return d

    def extract_capture_time(self) -> float:
        if (
            "GPS GPSDate" in self.tags
            and "GPS GPSTimeStamp" in self.tags  # Actually GPSDateStamp
        ):
            try:
                hours_f = get_tag_as_float(self.tags, "GPS GPSTimeStamp", 0)
                minutes_f = get_tag_as_float(self.tags, "GPS GPSTimeStamp", 1)
                if hours_f is None or minutes_f is None:
                    raise TypeError
                hours = int(hours_f)
                minutes = int(minutes_f)
                seconds = get_tag_as_float(self.tags, "GPS GPSTimeStamp", 2)
                gps_timestamp_string = "{0:s} {1:02d}:{2:02d}:{3:02f}".format(
                    self.tags["GPS GPSDate"].values, hours, minutes, seconds
                )
                return (
                    datetime.datetime.strptime(
                        gps_timestamp_string, "%Y:%m:%d %H:%M:%S.%f"
                    )
                    - datetime.datetime(1970, 1, 1)
                ).total_seconds()
            except (TypeError, ValueError):
                logger.info(
                    'The GPS time stamp in image file "{0:s}" is invalid. '
                    "Falling back to DateTime*".format(self.fileobj_name)
                )

        time_strings = [
            ("EXIF DateTimeOriginal", "EXIF SubSecTimeOriginal", "EXIF Tag 0x9011"),
            ("EXIF DateTimeDigitized", "EXIF SubSecTimeDigitized", "EXIF Tag 0x9012"),
            ("Image DateTime", "Image SubSecTime", "Image Tag 0x9010"),
        ]
        for datetime_tag, subsec_tag, offset_tag in time_strings:
            if datetime_tag in self.tags:
                date_time = self.tags[datetime_tag].values
                if subsec_tag in self.tags:
                    subsec_time = self.tags[subsec_tag].values
                else:
                    subsec_time = "0"
                try:
                    s = "{0:s}.{1:s}".format(date_time, subsec_time)
                    d = datetime.datetime.strptime(s, "%Y:%m:%d %H:%M:%S.%f")
                except ValueError:
                    logger.debug(
                        'The "{1:s}" time stamp or "{2:s}" tag is invalid in '
                        'image file "{0:s}"'.format(
                            self.fileobj_name, datetime_tag, subsec_tag
                        )
                    )
                    continue
                # Test for OffsetTimeOriginal | OffsetTimeDigitized | OffsetTime
                if offset_tag in self.tags:
                    offset_time = self.tags[offset_tag].values
                    try:
                        d += datetime.timedelta(
                            hours=-int(offset_time[0:3]), minutes=int(offset_time[4:6])
                        )
                    except (TypeError, ValueError):
                        logger.debug(
                            'The "{0:s}" time zone offset in image file "{1:s}"'
                            " is invalid".format(offset_tag, self.fileobj_name)
                        )
                        logger.debug(
                            'Naively assuming UTC on "{0:s}" in image file '
                            '"{1:s}"'.format(datetime_tag, self.fileobj_name)
                        )
                else:
                    logger.debug(
                        "No GPS time stamp and no time zone offset in image "
                        'file "{0:s}"'.format(self.fileobj_name)
                    )
                    logger.debug(
                        'Naively assuming UTC on "{0:s}" in image file "{1:s}"'.format(
                            datetime_tag, self.fileobj_name
                        )
                    )
                return (d - datetime.datetime(1970, 1, 1)).total_seconds()
        logger.info(
            'Image file "{0:s}" has no valid time stamp'.format(self.fileobj_name)
        )
        return 0.0
