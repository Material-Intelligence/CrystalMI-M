#include <Python.h>
#include <math.h>
#include <assert.h>

/*
#include "mathutils.h"
*/
#define Py_TRACE_REFS

#define MAX_NUM_POINTS_ON_PLANE 32
typedef double T_PLANE[4];
typedef double T_POINT[3];

static double vector_magnitude(const int len, const double *vec)
{
	int i;
	double res=0.0;
	for (i=0; i<len; ++i) {
		res += vec[i]*vec[i];
	}
	return sqrt(res);
}


static void cross_product(const double *v1, const double *v2, double *v)
{
	v[0] = v1[1]*v2[2] - v1[2]*v2[1];
	v[1] = v1[2]*v2[0] - v1[0]*v2[2];
	v[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


static double dot_product(const int len, const double *v1, const double *v2)
{
	int i;
	double res=0;
	for (i=0; i<len; ++i) {
		res += v1[i] * v2[i];
	}
	return res;
}


static double distance_point_plane(const double *point, const double *plane)
{
	/* given a plane equation a*x + b*y + c*z + d=0, calculates
	 * the point-plane distance from a point (x0, y0, z0) to this plane.
	 * The plane equation's coefficients a, b, c, and d are in the array
	 * plane, as {a, b, c, d}, where a^2+b^2+c^2==1. The distance has 
	 * a sign: it can be negative or positive or zero.
	 *
	 * Asserts:
	 * plane[0] * plane[0] + plane[1] * plane[1] + 
	 * plane[2] * plane[2] == 1; 
	 *
	 * See also:
	 * the function udistance_plane_point which calculates 
	 * unsigned distance.
	 * */
	double a=plane[0], b=plane[1], c=plane[2];
    // The following assert is commented out because it may fail
    // due to truncation error of floating-point numbers.
	//assert(a*a+b*b+c*c==1.0);
	return  a*point[0] + b*point[1] + c*point[2] + plane[3];
}


static double udistance_point_plane(const double *point, const double *plane)
{
	/* see the description of function distance_plane_point.
	 * This function always return a nonnegative double.
	 * */
	double distance = distance_point_plane(point, plane);
	if (distance < 0) distance = -distance;
	return distance;
}



static double distance_point_point(const double *point1, const double *point2)
{
	double delta_x, delta_y, delta_z;
	delta_x = point1[0] - point2[0];
	delta_y = point1[1] - point2[1];
	delta_z = point1[2] - point2[2];
	return sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
}


static double det3x3(const double *plane1, const double *plane2,
		     const double *plane3,
		     const int i, const int j, const int k)
{
	/* calculate the determinant of a 3x3 matrix. */
	return plane1[i] * (plane2[j]*plane3[k] - plane2[k]*plane3[j]) - 
	       plane1[j] * (plane2[i]*plane3[k] - plane2[k]*plane3[i]) +
	       plane1[k] * (plane2[i]*plane3[j] - plane2[j]*plane3[i]);
}

static int plane_intercept(const double *plane1, const double *plane2,
			   const double *plane3, double *point)
{
	/* given three planes (p1, p2, p3), calculates the interception point.
	 * This is solve an equation set.
	 * */
	double D;
	/* double Dx, Dy, Dz, D_reciprocal; */
	D = det3x3(plane1, plane2, plane3, 0, 1, 2);

	if(D==0.0) {  /* maybe abs(D)<epsilon is better */
		/* no solution, meaning the three planes has no interception.*/
		return -1; 
	}
	else {
		point[0] = -det3x3(plane1, plane2, plane3, 3, 1, 2) / D;
		point[1] = -det3x3(plane1, plane2, plane3, 0, 3, 2) / D;
		point[2] = -det3x3(plane1, plane2, plane3, 0, 1, 3) / D;
		return 0; /* found an interception point.*/
	}
}


static int point_within_enclosure(const int n_planes, const T_PLANE *planes,
				  const double *point, 
				  const int i, const int j, const int k)
{
	int index;
	double origin[] = {0.0, 0.0, 0.0};
	double d0, d;
	const double *pt_plane;
	for(index = 0; index < n_planes; ++index) {
		if((index == i) || (index == j) || (index == k)) continue;
		//pt_plane = &planes[index][0];
		pt_plane = planes[index];   // conversion from double[] to double *
		d  = distance_point_plane(point,  pt_plane);
		if(fabs(d)<1.0e-15) {
			continue; // this point is on the current plane.
		}
		d0 = distance_point_plane(origin, pt_plane);
		if (d * d0 < -1.0e-15) return 0;
		/* signed distances have different signs, so the the point
		 * resides at the other side of (0, 0, 0). */
	}
	return 1;
}


static int unique_point_in_set(const double *point, 
			       const int n_points, const T_POINT *points)
{
	/* whether a point is not overlaping with other points on a plane. 
	 * returns 0 if nonunique, 1 if unique (non-overlaping).
	 * */
	int i;
	double dist;
    double d0_r = 1.0 / sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
    if(d0_r<1.0) d0_r=1.0;  /*a small d0_r means large magntitude of length.*/

	for(i=0; i<n_points; ++i) {
		dist = distance_point_point(point, points[i]);
		/* if(dist == 0.0) return 0; */
		if(dist*d0_r < 1.0e-8) return 0;
	}
	return 1;
}


static int add_point(const double *pt, 
		     int *n_points, T_POINT **points, 
		     const int i, const int j, const int k)
{
	/* a helper function to add an point (pt) to the points array. 
	 * This does the following
	 * 1. check if the point being added is identical to any of the 
	 *    existant point; if so, the point is not going to be added.
	 * 2. check if the planes has more points than MAX_NUM_POINTS_ON_PLANE;
	 *    if so, I will throw out a warning, and not stop the program.
	 * The check for whether the point is at the same side of all other planes as
	 * the origin (0, 0, 0) was already done before calling this function. 
	 * */
	double *d;
	if(unique_point_in_set(pt, n_points[i], (const T_POINT *)points[i]) == 1) {
		d = points[i][n_points[i]];
		d[0]=pt[0]; d[1]=pt[1]; d[2]=pt[2];
		n_points[i] = n_points[i]+1;
	}

	if(unique_point_in_set(pt, n_points[j], (const T_POINT *)points[j]) == 1) {
		d = points[j][n_points[j]]; 
		d[0]=pt[0]; d[1]=pt[1]; d[2]=pt[2];
		n_points[j] = n_points[j]+1;
	}

	if(unique_point_in_set(pt, n_points[k], (const T_POINT *)points[k]) == 1) {
		d = points[k][n_points[k]]; 
		d[0]=pt[0]; d[1]=pt[1]; d[2]=pt[2];
		n_points[k] = n_points[k]+1;
	}

	if((n_points[i]>=MAX_NUM_POINTS_ON_PLANE) ||
	   (n_points[j]>=MAX_NUM_POINTS_ON_PLANE) ||
	   (n_points[k]>=MAX_NUM_POINTS_ON_PLANE)) {
		fprintf(stderr, 
			"Warning: maximum number of points on a plane has reached.\n"
			"    n_points[%d]=%d, n_points[%d]=%d, n_points[%d]=%d, and\n"
			"    MAX_NUM_POINTS_ON_PLANE=%d.\n"
	   	        "    Consider increasing MAX_NUM_POINTS_ON_PLANE.\n",
			i, n_points[i], j, n_points[j], k, n_points[k],
			MAX_NUM_POINTS_ON_PLANE);
	}
	return 0;
}



static int sort_points_by_convex(const int n_points, T_POINT *points)
{
	/* sort the points on a plane, so that the points form a convex polygon. 
	 * Assumes that the points are on a same plane, and that they indeed
	 * form a convex polygon. 
	 * */
	if(n_points<=3) return 1; /* no need to sort points less than 4. */
	int i, j, k;
	double d;
	/* create an index array for sorting. */
	int *idx = (int *) malloc(sizeof(int) * n_points);
	for(i=0; i<n_points; ++i) idx[i] = i;

	/* create an array of points of the same shape as points. */
	T_POINT *points_buf = malloc( sizeof(T_POINT) * n_points);
	
	double pt_array1[3];
	double pt_array2[3];

	for(i=1; i<n_points; ++i) {
		for(j=0; j<3; ++j) {
			points_buf[i][j] = points[i][j] - points[0][j];
		}
	}
	
	/* assume an edge of the convex polygon is idex[0]---idx[1]. */
	cross_product(points_buf[2], points_buf[1], pt_array1);
	for(i=n_points-1; i>=3; --i) {
		cross_product(points_buf[idx[i]], points_buf[idx[1]], 
			      pt_array2);
		d = dot_product(3, pt_array1, pt_array2);
		if(d<0) { 
			/* points i and i-1 are at different sides of idx[0]---idx[1].*/
			j = idx[i];
			idx[i] = idx[1];
			idx[1] = j;
			cross_product(points_buf[idx[2]], points_buf[idx[1]], 
				      pt_array1);
		}
	}
	/* now idx[0]---idx[1] is a valid edge of a convex polygon. */
	
	for(k=2; k<n_points-1; ++k) {
		for(i=0; i<n_points; ++i) {
			for(j=0; j<3; ++j) {
				points_buf[idx[i]][j] = points[idx[i]][j] -
				                   points[idx[k-1]][j];
			}
		}
		cross_product(points_buf[idx[k-2]], points_buf[idx[k]], 
			      pt_array1);
		for(j=n_points-1; j>=k+1; --j) {
			cross_product(points_buf[idx[j]], points_buf[idx[k]], 
				      pt_array2);
			d = dot_product(3, pt_array1, pt_array2);
			if(d<0) {
				i = idx[j];
				idx[j] = idx[k];
				idx[k] = i;
				cross_product(points_buf[idx[k-2]], 
					      points_buf[i], pt_array1);
			}
		}
	}

	/* make a copy of the points */
	/*
	for(i=0; i<n_points; ++i) {
		for(j=0; j<3; ++j) {
			points_buf[i][j] = points[i][j];
		}
	}
	*/
	double *d_pt_buf=points_buf[0];
	double *d_pt=points[0];
	for(i=0; i<n_points*3; ++i, ++d_pt_buf, ++d_pt) {
		*d_pt_buf = *d_pt;
	}

	/* then copy back according to the index. */
	for(i=0; i<n_points; ++i) {
		j = idx[i];
		if(j != i) {
			points[i][0] = points_buf[j][0];
			points[i][1] = points_buf[j][1];
			points[i][2] = points_buf[j][2];
		}
	}

	/* free the memory allocated in this function call. */
	free(points_buf);
	free(idx);

	return 0;
}


static double area_coplanar(const int n_points, const T_POINT *points)
{
	if(n_points<3) return 0.0;
	int i;
	double a, b, c, s;
	double area=0.0;
	// sort_points_by_convex(plane, n_points, points);
	a = distance_point_point(points[0], points[1]);
	for(i=1; i<n_points-1; ++i) {
		b = distance_point_point(points[i], points[i+1]);
		c = distance_point_point(points[0], points[i+1]);
		s = (a+b+c)/2.0;
		area = area + sqrt(s*(s-a)*(s-b)*(s-c));
		a = c;
		//d=triangle_area_by_coordinate(points[0], points[i+1], points[i+2]);
		//area += d;
	}
	return area;
}


static double volume_enclosure(const int n_planes, const T_PLANE *planes,
			       const double *areas) 
{
	double origin[3] = {0.0, 0.0, 0.0};
	int i;
	double dist;
	double v=0;
	for(i=0; i<n_planes; ++i) {
		if(areas[i]==0.0) continue;
		dist = udistance_point_plane(origin, planes[i] );
		v = v + 1.0/3.0 * dist * areas[i];
	}
	return v;
}


static int enclosure(const int n_planes, const T_PLANE *planes,
		     int *n_points, T_POINT **points, 
		     double *areas, double *volume)
{
	/* calculates the geometric parameters enclosed by the planes.
	 * Inputs: 
	 * n_planes (const int): number of planes;
	 * planes (const T_PLANE *): n_planes x 4 array of double.
	 * Outpus:
	 * n_points (int *): array of number of interception points on each plane;
	 * points (T_POINT **): n_planes x n_points x 3 array of double;
	 * areas (double *): areas of enclosed part on each plane;
	 * volume (double *): the volume of enclosed space.
	 * */
	if(n_planes < 3) {
		printf("At least 4 planes can form an enclosed space.\n");
		return -1;
	}
	int i, j, k, r, w;
	double pt[3] = {0.0, 0.0, 0.0};

	/* reset the number of points of the planes to zero. */
	for(i=0; i<n_planes; ++i) {
		n_points[i] = 0;
	}

	for(i=0; i<n_planes-2; ++i) {
		for(j=i+1; j<n_planes-1; ++j) {
			for(k=j+1; k<n_planes; ++k) {
				r = plane_intercept(planes[i], planes[j], 
						    planes[k], pt);
				/* r may be -1 or 0, for which -1 means no
				 * interception point, and 0 means an 
				 * interception point. 
				 * Futher check if the point is within the 
				 * enclosure, that is, it resides at the same 
				 * side of all the planes as the origin
				 * (0, 0, 0)
				 * */
				if(r!=0) {
					/* no interception. */
					continue;
				}
				w=point_within_enclosure(n_planes, planes, pt, i, j, k);
				if(w==1) {
					add_point(pt, n_points, points, i, j, k);
				}
			}
		}
	}
	/* The points are ready now. The next step is to calculate the areas.
	 * I need to sort the points on every plane first.
	 * */
	for(i=0; i<n_planes; ++i) {
		sort_points_by_convex(n_points[i], points[i]);
		areas[i] = area_coplanar(n_points[i], (const T_POINT *)points[i]);
	}
	*volume = volume_enclosure(n_planes, planes, areas);
	return 0;
}


static int get4f_parse(PyObject *obj, double *plane) {
	int i;
	PyObject *po_temp;
	for(i=0; i<4; ++i) {
		po_temp = PyList_GetItem(obj, i);
		if(PyFloat_Check(po_temp)) {
			plane[i] = PyFloat_AsDouble(po_temp);
		}
		else if(PyLong_Check(po_temp)) {
			plane[i] = PyLong_AsLong(po_temp);
		}
		else {
			fprintf(stderr, 
				"Failed to parse a non-number as a number.\n"
				"    The value being passed is: ");
			PyObject_Print(po_temp, stderr, Py_PRINT_RAW);
			fprintf(stderr, "\n");
			return -1;
		}
	}
	return 0;
}


static PyObject *
enclosure_py(PyObject *self, PyObject *args) {
	PyObject *po_arg;
	if(!PyArg_ParseTuple(args, "O", &po_arg)) {
		printf("Incorrect argument. Expecting ONE tuple or list.\n");
		return NULL;
	}
	int n_planes;
	T_PLANE *planes;

	int i, j;
	PyObject *po_temp;
	if(PyTuple_Check(po_arg)) {
		//printf("It is a tuple.\n");
		n_planes = PyTuple_Size(po_arg);
		//printf("The size of the tuple is %d.\n", n_planes);
		planes = malloc(n_planes * sizeof *planes);
		//assert(planes);
		if(!planes) {
			fprintf(stderr, "Error: Insufficient memory. Exiting...\n");
			exit(1);
		}
		for(i=0; i<n_planes; ++i) {
			po_temp = PyTuple_GetItem(po_arg, i);
			get4f_parse(po_temp, planes[i]);
			// PyArg_ParseTuple(po_temp, "[dddd]", 
			// 	planes[i], planes[i]+1, planes[i]+2, planes[i]+3);
			// Py_DECREF(po_temp);
			// po_temp contains a borrowed reference to po_arg[i], so 
			// it is not responsible for reference counting.
		}
	}
	else if(PyList_Check(po_arg)) {
		//printf("It is a list.\n");
		n_planes = PyList_Size(po_arg);
		//printf("The size of the list is %d.\n", n_planes);
		planes = malloc(n_planes * sizeof *planes);
		for(i=0; i<n_planes; ++i) {
			po_temp = PyList_GetItem(po_arg, i);
			get4f_parse(po_temp, planes[i]);
			// PyArg_ParseList(po_temp, "[d,d,d,d]", planes[i],
			// 	planes[i]+1, planes[i]+2, planes[i]+3);
			// Py_DECREF(po_temp);
		}
	}
	else {
		fprintf(stderr, "It is neither a list nor tuple. I am stuck.\n");
		return NULL;
	}

	/*
	for(i=0; i<n_planes; ++i){
		for(j=0; j<4; ++j){
			printf("planes[%d][%d]=%f\n", i, j, planes[i][j]);
		}
	}
	*/


	int *n_points = malloc(n_planes * sizeof *n_points);
	T_POINT **points = malloc(n_planes * sizeof *points);
	assert(points);
	for(i=0; i<n_planes; ++i) {
		points[i] = malloc(MAX_NUM_POINTS_ON_PLANE * sizeof(T_POINT));
		assert(points[i]);
	}

	double *areas = malloc(sizeof(double) * n_planes);
	double volume=0.0;

	const T_PLANE *cplanes = (const T_PLANE *)planes;
	enclosure(n_planes, cplanes, n_points, points, areas, &volume);

	/*
	for(i=0; i<n_planes; ++i) {
		printf("areas[%d] is %f, n_points[%d] is %d.\n", i, areas[i], i, n_points[i]);
		for(j=0; j<n_points[i]; ++j){
			printf("    [%f, %f, %f]\n", points[i][j][0], points[i][j][1], points[i][j][2]);
		}
		printf("\n");
	}
	*/

	// now build the return value.
	PyObject *po_points;
	PyObject *po_item;
	po_points = PyList_New(n_planes);
	for(i=0; i<n_planes; ++i) {
		po_item = PyList_New(n_points[i]);
		for(j=0; j<n_points[i]; ++j) {
			PyList_SetItem(po_item, j, 
				Py_BuildValue("[ddd]", points[i][j][0],
				points[i][j][1], points[i][j][2]));
		}
		PyList_SetItem(po_points, i, po_item);
		// Py_DECREF(po_item);
	}
	
	PyObject *po_areas;
	po_areas = PyList_New(n_planes);
	for(i=0; i<n_planes; ++i) {
		PyList_SetItem(po_areas, i, PyFloat_FromDouble(areas[i]));
	}

	PyObject *po_volume = PyFloat_FromDouble(volume);

	Py_INCREF(po_volume);
	Py_INCREF(po_areas);
	Py_INCREF(po_points);

	PyObject *po_return;
	po_return = Py_BuildValue("[OOO]", po_volume, po_areas, po_points);

	// free the malloc'ed memory.
	free(areas);
	free(n_points);
	for(i=0; i<n_planes; ++i) {
		free(points[i]);
	}
	free(points);
	free(planes);

	Py_INCREF(po_return);
	return po_return;
}


// Method definition object
static PyMethodDef CGeometryMethods[] = {
    {"enclosure", enclosure_py, METH_VARARGS,
     "Calculate the surface area and volume of an enclosed space by planes."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef cgeometrymodule = {
    PyModuleDef_HEAD_INIT,
    "cgeometry",   // Module name
    NULL,          // Module documentation, may be NULL
    -1,            // Size of per-interpreter state of the module
    CGeometryMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_cgeometry(void)
{
    return PyModule_Create(&cgeometrymodule);
}

