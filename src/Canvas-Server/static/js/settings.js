
export const CANVAS_ELEMENT_ID = "drawing-canvas";

export const INITIAL_UNITS_TO_PIXELS = 300;
export const INITIAL_CAMERA_POSITION = { x: 0, y: 0 };

export const INITIAL_START_POINT = { x: 0, y: 0 };
export const INITIAL_TARGET_POINT = { x: 1, y: 0 };

export const INITIAL_POLYGONS = [
	[{ x: 0.5, y: 0.5 }, { x: 1.5, y: 0.5 }, { x: 1, y: 1 }],
	[{ x: -0.5, y: 0.0 }, { x: -1.5, y: -0.5 }, { x: -1, y: -1 }, { x: -0.5, y: -1 }, { x: 0.5, y: -0.5 }],
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

// Grid lines are always at most `MINIMUM_GRID_SPACING` pixels apart, 
// but they can be farther apart depending on the zoom level. 
// This ensures that the grid is always visible and not too cluttered.
export const MINIMUM_GRID_SPACING = 83;

export const HIT_RADIUS = 15;
export const SCROLL_SENSITIVITY = 0.0005;
export const DOUBLE_CLICK_TIME = 300; // milliseconds

export const SNAP_BUTTON_ID = "snapping-toggle";
export const MAKE_TRIANGLE_BUTTON_ID = "triangle-button";
export const SHOW_VERTEX_LINE_BUTTON_ID = "vertex-line-toggle";
export const SAVE_DRAWING_BUTTON_ID = "save-drawing-final-button";

export const OVERLAY_ELEMENTS_ID = [
	"saved-drawings-overlay",
	"save-drawing-overlay",
	"user-login-section-outer",
	"save-prompt",
	"rename-prompt",
	"duplicate-prompt",
];

export const BLOCK_CLICK_IDS = [
	"dropdown-content",
	"save-dropdown-menu",
]

export const SAVE_DRAWING_ENDPOINT_URL = "/drawings/save";
export const UPDATE_DRAWING_ENDPOINT_URL = "/drawings";
