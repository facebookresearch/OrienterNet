scenes = {
    "seattle": [
        "reloc_seattle_downtown",
        "reloc_seattle_pike",
        "reloc_seattle_westlake",
    ],
    "detroit": ["reloc_detroit_greektown", "reloc_detroit_gcp"],
}

data_cfg = {
    "dump_dir": "manifold://psarlin/tree/maploc/data/aria_v2_aligned",
    "max_init_error": 0,
    "init_from_gps": False,
    "return_gps": True,
    "filter_for": None,
    "target_focal_length": 256,
    "pixel_per_meter": 2,
    "crop_size_meters": 64,
}
