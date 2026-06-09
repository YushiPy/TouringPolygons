
export const CANVAS_ELEMENT_ID = "drawing-canvas";

export const INITIAL_UNITS_TO_PIXELS = 266.07748625964655;

export const INITIAL_CAMERA_POSITION = {
	x: 0.8642425202470306,
	y: 0.7865682930827428,
};

export const INITIAL_START_POINT = {
	x: -0.2,
	y: 0.7000000000000001,
};

export const INITIAL_TARGET_POINT = {
	x: 1.1,
	y: 0.2,
};

export const INITIAL_POLYGONS = [
	[
		{ x: 0, y: 1.5 },
		{ x: -0.4, y: 1.48 },
		{ x: -0.9, y: 1.48 },
		{ x: -1.3, y: 1.5 },
		{ x: -1.3, y: 1.3 },
		{ x: -0.74, y: 1.3 },
		{ x: -0.72, y: 0.5 },
		{ x: -0.74, y: 0 },
		{ x: -0.54, y: 0 },
		{ x: -0.56, y: 0.5 },
		{ x: -0.54, y: 1.3 },
		{ x: 0, y: 1.3 },
	],
	[
		{ x: 1.4000000000000001, y: 1.5 },
		{ x: 0.4, y: 1.5 },
		{ x: 0.4, y: 0 },
		{ x: 0.6000000000000001, y: 0 },
		{ x: 0.6000000000000001, y: 1.3 },
		{ x: 1.2000000000000002, y: 1.3 },
		{ x: 1.2000000000000002, y: 0.8 },
		{ x: 0.7000000000000001, y: 0.8 },
		{ x: 0.7000000000000001, y: 0.6000000000000001 },
		{ x: 1.4000000000000001, y: 0.6000000000000001 },
	],
	[
		{ x: 2.8000000000000003, y: 1.5 },
		{ x: 2.8000000000000003, y: 0.6000000000000001 },
		{ x: 2.2, y: 0.6000000000000001 },
		{ x: 2.2, y: 0.8 },
		{ x: 2.6, y: 0.8 },
		{ x: 2.6, y: 1.3 },
		{ x: 2, y: 1.3 },
		{ x: 2, y: 0 },
		{ x: 1.8, y: 0 },
		{ x: 1.8, y: 1.5 },
	],
];

export const START_POINT_COLOR = "#00FFFF";
export const TARGET_POINT_COLOR = "#FF00FF";

export const POLYGON_COLORS = [
	"#FFFF00",
	"#0000FF",
	"#00FF00",
	"#FF0000",
	"#00FFFF",
	"#FF00FF"
]

export const POLYGON_INSIDE_ALPHA = 0.25;

export const SOLUTION_COLOR = "#DC9D2F";

export const BACKGROUND_COLOR = "#121212";
export const POINT_RADIUS = 8;

export const MAIN_AXIS_COLOR = "#E0E0E0";
export const GRID_COLOR = "#555555";
export const SUB_GRID_COLOR = "#2A2A2A";

export const GRID_NUMBER_COLOR = "#E0E0E0";
export const GRID_NUMBER_LIGHT_COLOR = "#777777";

export const GRID_NUMBER_FONT = "14px Arial";

export const MAIN_AXIS_WIDTH = 1;
export const GRID_WIDTH = 1;
export const SUB_GRID_WIDTH = 1;

export const MINIMUM_GRID_SPACING = 83;

export const HIT_RADIUS = 15;
export const SCROLL_SENSITIVITY = 0.0005;
export const DOUBLE_CLICK_TIME = 300;

export const SNAP_BUTTON_ID = "snapping-toggle";
export const MAKE_TRIANGLE_BUTTON_ID = "triangle-button";
export const SHOW_VERTEX_LINE_BUTTON_ID = "vertex-line-toggle";
export const MAP_BUTTON_ID = "map-toggle";

export const OVERLAY_ELEMENTS_ID = [
	"saved-drawings-overlay",
	"save-prompt",
	"rename-prompt",
	"duplicate-prompt",
];

export const BLOCK_CLICK_IDS = [
	"dropdown-content",
	"save-dropdown-menu",
]
