// Define label tags

// Object tags
#define ARGUMENTS_TAG 1
#define RESULTS_TAG 2

// Function tags
#define FUNCTION_TAG 1


// Center tags
#define X_CENTER_TAG 1
#define Y_CENTER_TAG 2
#define Z_CENTER_TAG 3


//Normal arguments tags
#define NORMAL_ARG_BASE 2
#define NORMAL_ARG_PNT  3


// Box arguments tags
#define BOX_ARG_X  1
#define BOX_ARG_Y  2
#define BOX_ARG_Z  3
#define BOX_ARG_DX 4
#define BOX_ARG_DY 5
#define BOX_ARG_DZ 6
#define BOX_ARG_PNT1 7
#define BOX_ARG_PNT2 8



// Cylinder arguments tags
#define CYLINDER_ARG_CENTER 2
#define CYLINDER_ARG_VECTOR 3
#define CYLINDER_ARG_RADIUS 4
#define CYLINDER_ARG_HEIGHT 5

// Sphere arguments tags
#define SPHERE_ARG_CENTER 2
#define SPHERE_ARG_RADIUS 3

// Cone arguments tags
#define CONE_ARG_R1  1
#define CONE_ARG_R2  2
#define CONE_ARG_H   3
#define CONE_ARG_POINT 4
#define CONE_ARG_VECTOR 5

// Torus arguments tags
#define TORUS_ARG_CENTER   1
#define TORUS_ARG_VECTOR   2
#define TORUS_ARG_RMAJOR   3
#define TORUS_ARG_RMINOR   4

// Boolean operation arguments tags
#define BOOL_ARG_SHAPE1 1
#define BOOL_ARG_SHAPE2 2
#define MULTIFUSE_TAG 1
#define MULTICUT_TAG 1

#define BOOL_ARG_SELFINTERSECTION 3
#define BOOL_ARG_REMOVEEXTRAEDGES 4
#define BOOL_ARG_TOLERANCE 5


// Translate operation arguments tags
#define TRANSLATE_ARG_MOVE     1
#define TRANSLATE_ARG_CONTEXT  2
#define TRANSLATE_ARG_ORIGINAL 3
#define TRANSLATE_ARG_DX       4
#define TRANSLATE_ARG_DY       5
#define TRANSLATE_ARG_DZ       6
#define TRANSLATE_ARG_POINT1   7
#define TRANSLATE_ARG_POINT2   8
#define TRANSLATE_ARG_VECTOR   9
#define TRANSLATE_ARG_DISTANCE 10
#define TRANSLATE_ARG_COPYMODE 11



// Rotate operation arguments tags
#define ROTATE_ARG_MOVE     1
#define ROTATE_ARG_CONTEXT  2
#define ROTATE_ARG_ORIGINAL 3
#define ROTATE_ARG_CENTER   4
#define ROTATE_ARG_POINT1   5
#define ROTATE_ARG_POINT2   6
#define ROTATE_ARG_AXIS     7
#define ROTATE_ARG_ANGLE    8
#define ROTATE_ARG_COPYMODE 9



// MultiRotate operation arguments tags
#define MULTIROTATE_ARG_CONTEXT  1
#define MULTIROTATE_ARG_AXIS     2
#define MULTIROTATE_ARG_ANGLE    3
#define MULTIROTATE_ARG_NUM      4



// Mirror operation arguments tags
#define MIRROR_ARG_MOVE     1
#define MIRROR_ARG_CONTEXT  2
#define MIRROR_ARG_ORIGINAL 3
#define MIRROR_ARG_POINT    4
#define MIRROR_ARG_AXIS     5
#define MIRROR_ARG_PLANE    6
#define MIRROR_ARG_COPYMODE 7



// Prism arguments tags
#define PRISM_ARG_HEIGHT 1
#define PRISM_ARG_BASIS 2
#define PRISM_ARG_VECTOR 3
#define PRISM_ARG_REVERSED 4




//Selection
#define SELECTION_CONTEXT_TAG 1



// Transition modes for pipe shell

#define TRANSFORM_TRANSITION_MODE	0

#define RIGTHCORNER_TRANSITION_MODE	1

#define ROUNDCORNER_TRANSITION_MODE	2


//Error messages
#define DONE                  "Done"

#define NOTDONE               "Can't be done"
#define NULL_ACCESS_NODE      "Null access node"
#define NULL_BASIS_NODE       "Null basis node"
#define NULL_ARGUMENT         "Null argument" 
#define NO_ROOT_FOUND         "No root of the tree found"
#define ALGO_FAILED           "Algorithm failed"
#define UNABLE_DUE_TO_CYCLE   "Unable to set arguments, may be a cyclic dependence"
#define UNABLE_DUE_TO_NAMING  "Unable to set arguments, may be a wrong naming"







//Build Shape Orientation tag
#define ORIENTATION_TAG  1



//Circle arguments tags
#define CIRCLE_ARG_POINT1  2
#define CIRCLE_ARG_POINT2  3
#define CIRCLE_ARG_POINT3  4

#define CIRCLE_ARG_CENTER  5
#define CIRCLE_ARG_VECTOR  6
#define CIRCLE_ARG_RADIUS  7


//Ellipse arguments tags
#define ELLIPSE_ARG_CENTER       2
#define ELLIPSE_ARG_VECTOR       3
#define ELLIPSE_ARG_VECTORMAJOR  4
#define ELLIPSE_ARG_RMAJOR       5
#define ELLIPSE_ARG_RMINOR       6


//Parabola arguments tags
#define PARABOLA_ARG_CENTER       2
#define PARABOLA_ARG_VECTOR       3
#define PARABOLA_ARG_VECTORMAJOR  4
#define PARABOLA_ARG_FOCAL        5
#define PARABOLA_ARG_PARAMT1      6
#define PARABOLA_ARG_PARAMT2      7


//Arc arguments tags
#define ARC_ARG_POINT1       2
#define ARC_ARG_POINT2       3
#define ARC_ARG_POINT3       4
#define ARC_ARG_SENSE        5



//Plane arguments tags
#define PLANE_ARG_SIZE 1

#define PLANE_ARG_POINT1 2
#define PLANE_ARG_POINT2 3
#define PLANE_ARG_POINT3 4

#define PLANE_ARG_VECTOR 5

#define PLANE_ARG_FACE 6

#define PLANE_ARG_PARAM_U 7

#define PLANE_ARG_PARAM_V 8

#define PLANE_ARG_VECTOR1    9
#define PLANE_ARG_VECTOR2    10

#define PLANE_ARG_ORIENT  11

#define PLANE_ARG_LCS     12


//Vector arguments tags
#define VECTOR_ARG_POINT1 2
#define VECTOR_ARG_POINT2 3

#define VECTOR_ARG_DX 2
#define VECTOR_ARG_DY 3
#define VECTOR_ARG_DZ 4

#define VECTOR_ARG_CURVE   2
#define VECTOR_ARG_T 3

#define VECTOR_ARG_SURFACE 2
#define VECTOR_ARG_U 3
#define VECTOR_ARG_V 4
#define VECTOR_ARG_PNT 5



//Vertex Edge arguments tags
#define VERTEX_ARG_X     1
#define VERTEX_ARG_Y     2
#define VERTEX_ARG_Z     3

#define VERTEX_ARG_REF   4

#define VERTEX_ARG_PARAM_T 5
#define VERTEX_ARG_CURVE 6
#define VERTEX_ARG_LINE1 7
#define VERTEX_ARG_LINE2 8 

#define VERTEX_ARG_SURFACE 9
#define VERTEX_ARG_PARAM_U 10
#define VERTEX_ARG_PARAM_V 11

//Edge arguments tags
#define EDGE_ARG_POINT1 2
#define EDGE_ARG_POINT2 3

#define EDGE_ARG_LINE 4

#define EDGE_ARG_SURFACE 2
#define EDGE_ARG_PARAM_U1 3
#define EDGE_ARG_PARAM_V1 4
#define EDGE_ARG_PARAM_U2 5
#define EDGE_ARG_PARAM_V2 6
#define EDGE_ARG_PARAM_T1 7
#define EDGE_ARG_PARAM_T2 8

//Wire arguments tags
#define WIRE_BUILD_TOLERANCE_TAG 2
#define WIRE_BUILD_TAG 3

//Face arguments tags
#define FACE_BUILD_FACEARG_TAG 2
#define FACE_BUILD_ISPLANAR_TAG 3
#define FACE_BUILD_WIREARG_TAG 4

//Shell arguments tags
#define SHELL_BUILD_TAG 2

//Solid arguments tags
#define SOLID_BUILD_TAG 1


//Thrusection arguments tags
#define THRUSECTION_BUILD_TOLERANCE_TAG 2
#define THRUSECTION_BUILD_ISSOLID_TAG   3
#define THRUSECTION_BUILD_ISRULED_TAG   4
#define THRUSECTION_BUILD_SECTION_TAG   5


// Pipe shell arguments tags
#define PIPESHELL_IS_SOLID_TAG                  1
#define PIPESHELL_IS_FRENET_TAG	                2
#define PIPESHEL_TRANSITION_MODE_TAG            3
#define PIPESHELL_SPINE_TAG                     4
#define PIPESHELL_MODE_TAG                      5
#define PIPESHELL_PROFILE_TAG                   6

//Revolution tags
#define REVOL_ARG_ANGLE    1
#define REVOL_ARG_AXIS     2
#define REVOL_ARG_BASE     3

//Polygon arguments tags
#define POLYGON_BUILD_TAG 2
#define POLYGON_CLOSE_TAG 3



//CosPeriodEdge arguments tags
#define CPE_PARAM_R	        2
#define CPE_PARAM_RPL		3
#define CPE_PARAM_RD		4
#define CPE_PARAM_AXISDIR	5
#define CPE_PARAM_AMPDIR	6
#define CPE_PARAM_OX		7
#define CPE_PARAM_OY		8
#define CPE_PARAM_OZ		9
#define CPE_PARAM_PERIODNUM	10
#define CPE_PARAM_PHASE 	11
#define CPE_PARAM_ONEPERIODSAMPLENUM	12

//CosPeriodEdge arguments tags
#define RPE_PARAM_R	        2
#define RPE_PARAM_RD		3
#define RPE_PARAM_AL		4
#define RPE_PARAM_ZL		5
#define RPE_PARAM_AXISDIR	6
#define RPE_PARAM_AMPDIR	7
#define RPE_PARAM_OX		8
#define RPE_PARAM_OY		9
#define RPE_PARAM_OZ		10
#define RPE_PARAM_PERIODNUM	11


//HelixEdge arguments tags
#define HELIX_PARAM_OX		2
#define HELIX_PARAM_OY		3
#define HELIX_PARAM_OZ		4
#define HELIX_PARAM_R	        5
#define HELIX_PARAM_L		6
#define HELIX_PARAM_AXISDIR	7
#define HELIX_PARAM_PERIODNUM	8


// Pipe arguments tags
#define PIPE_SPINE_TAG 2
#define PIPE_PROFILE_TAG 3


// Translate operation arguments tags
#define PERIODSHAPE_ARG_CONTEXT  1
#define PERIODSHAPE_ARG_DX       2
#define PERIODSHAPE_ARG_DY       3
#define PERIODSHAPE_ARG_DZ       4
#define PERIODSHAPE_ARG_NUM      5


//Curve arguments tags
#define CURVE_ROWCOUNT  1
#define CURVE_TABLE_DATA  2
#define CURVE_ARRAY	3

