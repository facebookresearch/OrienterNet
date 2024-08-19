from omegaconf import OmegaConf

from ...utils.geo import BoundaryBox

location_to_params = {
    "sanfrancisco_soma": {
        "bbox": BoundaryBox((37.770364, -122.410307), (37.795545, -122.388772)),
        "bbox_val": BoundaryBox(
            (37.788123419925945, -122.40053535863909),
            (37.78897443253716, -122.3994618718349),
        ),
        "filters": {"model": "GoPro Max"},
        "osm_file": "sanfrancisco.osm",
    },
    "sanfrancisco_hayes": {
        "bbox": BoundaryBox((37.768634, -122.438415), (37.783894, -122.410605)),
        "bbox_val": BoundaryBox(
            (37.77682908567614, -122.42439593370665),
            (37.7776996640339, -122.42329849537967),
        ),
        "filters": {"model": "GoPro Max"},
        "osm_file": "sanfrancisco.osm",
    },
    "montrouge": {
        "bbox": BoundaryBox((48.80874, 2.298958), (48.825276, 2.332989)),
        "bbox_val": BoundaryBox(
            (48.81554465300679, 2.315590378986898),
            (48.816228935240346, 2.3166087395920103),
        ),
        "filters": {"model": "LG-R105"},
        "osm_file": "paris.osm",
    },
    "amsterdam": {
        "bbox": BoundaryBox((52.340679, 4.845284), (52.386299, 4.926147)),
        "bbox_val": BoundaryBox(
            (52.358275965541495, 4.876867175817335),
            (52.35920971624303, 4.878370977965195),
        ),
        "filters": {"model": "GoPro Max"},
        "osm_file": "amsterdam.osm",
    },
    "lemans": {
        "bbox": BoundaryBox((47.995125, 0.185752), (48.014209, 0.224088)),
        "bbox_val": BoundaryBox(
            (48.00468200256593, 0.20130905922712253),
            (48.00555356009431, 0.20251886369476968),
        ),
        "filters": {"creator_username": "sogefi"},
        "osm_file": "lemans.osm",
    },
    "berlin": {
        "bbox": BoundaryBox((52.459656, 13.416271), (52.499195, 13.469829)),
        "bbox_val": BoundaryBox(
            (52.47478263625299, 13.436060761632277),
            (52.47610554128314, 13.438407628895831),
        ),
        "filters": {"is_pano": True},
        "osm_file": "berlin.osm",
    },
    "nantes": {
        "bbox": BoundaryBox((47.198289, -1.585839), (47.236161, -1.51318)),
        "bbox_val": BoundaryBox(
            (47.212224982547106, -1.555772859366718),
            (47.213374064189956, -1.554270622470525),
        ),
        "filters": {"is_pano": True},
        "osm_file": "nantes.osm",
    },
    "toulouse": {
        "bbox": BoundaryBox((43.591434, 1.429457), (43.61343, 1.456653)),
        "bbox_val": BoundaryBox(
            (43.60314813839066, 1.4431497839062253),
            (43.604433961018984, 1.4448508228862122),
        ),
        "filters": {"is_pano": True},
        "osm_file": "toulouse.osm",
    },
    "vilnius": {
        "bbox": BoundaryBox((54.672956, 25.258633), (54.696755, 25.296094)),
        "bbox_val": BoundaryBox(
            (54.68292611300143, 25.276979025529165),
            (54.68349008447563, 25.27798847871685),
        ),
        "filters": {"is_pano": True},
        "osm_file": "vilnius.osm",
    },
    "helsinki": {
        "bbox": BoundaryBox(
            (60.1449128318, 24.8975480117), (60.1770977471, 24.9816543235)
        ),
        "bbox_val": BoundaryBox(
            (60.163825618884275, 24.930182541064955),
            (60.16518598734065, 24.93274647451007),
        ),
        "filters": {"is_pano": True},
        "osm_file": "helsinki.osm",
    },
    "milan": {
        "bbox": BoundaryBox(
            (45.4810977947, 9.1732723899), (45.5284238563, 9.2255987917)
        ),
        "bbox_val": BoundaryBox(
            (45.502686834500466, 9.189078329923374),
            (45.50329294217317, 9.189881944589828),
        ),
        "filters": {"is_pano": True},
        "osm_file": "milan.osm",
    },
    "avignon": {
        "bbox": BoundaryBox(
            (43.9416178156, 4.7887045302), (43.9584848909, 4.8227015622)
        ),
        "bbox_val": BoundaryBox(
            (43.94768786305171, 4.809099008430249),
            (43.94827840894793, 4.809954737764413),
        ),
        "filters": {"is_pano": True},
        "osm_file": "avignon.osm",
    },
    "paris": {
        "bbox": BoundaryBox((48.833827, 2.306823), (48.889335, 2.39067)),
        "bbox_val": BoundaryBox(
            (48.85558288211851, 2.3427920762801526),
            (48.85703370256603, 2.3449544861818654),
        ),
        "filters": {"is_pano": True},
        "osm_file": "paris.osm",
    },
}

default_cfg = OmegaConf.create(
    {
        "downsampling_resolution_meters": 2,
        "target_num_val_images": 100,
        "val_train_margin_meters": 25,
        "max_num_train_images": 50_000,
        "max_image_size": 512,
        "do_legacy_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)
