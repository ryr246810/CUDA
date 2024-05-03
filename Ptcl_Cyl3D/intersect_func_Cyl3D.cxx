#include <intersect_func_Cyl3D.cuh>
#include <TxVector2D.h>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

bool detect_intersect_Cyl3D(const TxVector2D<double>& a,  const TxVector2D<double>& b, const TxVector2D<double>& c,  const TxVector2D<double>& d){
	//1、快速排斥
	bool is_overlapped = min(a[0], b[0]) <= max(c[0], d[0])
					  && min(a[1], b[1]) <= max(c[1], d[1])
					  && min(c[0], d[0]) <= max(a[0], b[0])
					  && min(c[1], d[1]) <= max(a[1], b[1]);
	
	//cout<<"overlapped state = "<<is_overlapped<<endl;
	
	if(!is_overlapped){
		return false;
	}
	
	//2、跨立实验
	TxVector2D<double> ac = c - a;
	TxVector2D<double> ad = d - a;
	TxVector2D<double> ab = b - a;
	
	double u = ac[0] * ab[1] - ac[1] * ab[0]; // ac * ab
	double v = ad[0] * ab[1] - ad[1] * ab[0]; // ad * ab
	
	TxVector2D<double> ca = a - c;
	TxVector2D<double> cb = b - c;
	TxVector2D<double> cd = d - c;
	
	double w = ca[0] * cd[1] - ca[1] * cd[0]; // ca * cd
	double z = cb[0] * cd[1] - cb[1] * cd[0]; // cb * cd
	
	if( u * v <= 0 && w * z <= 0){
		return true;
	}
	else{
		//cout<<"final return false"<<endl;
		//cout<<"u * v = "<<(u*v)<<endl;
		//cout<<"w * z = "<<(w*z)<<endl;
		return false;
	}
}

// TxVector2D<double> get_intersect_pnt(const TxVector2D<double>& a,  const TxVector2D<double>& b, const TxVector2D<double>& c,  const TxVector2D<double>& d){
	// double x1 = a[0];
	// double y1 = a[1];
	// double x2 = b[0];
	// double y2 = b[1];

	// double x3 = c[0];
	// double y3 = c[1];
	// double x4 = d[0];
	// double y4 = d[1];	

	// double b1 = (y2 - y1) * x1 + (x1 - x2) * y1;
	// double b2 = (y4 - y3) * x3 + (x1 - x2) * y3;
	// double D = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1);
	// double D1 = b2 * (x2 - x1) - b1 * (x4 - x3);
	// double D2 = b2 * (y2 - y1) - b1 * (y4 - y3);
	
	// double x0 = D1 / D;
	// double y0 = D2 / D;
	// return TxVector2D<double>(x0, y0);
// }

TxVector2D<double> get_intersect_pnt_Cyl3D(const TxVector2D<double>& a1,  const TxVector2D<double>& a2, const TxVector2D<double>& b1,  const TxVector2D<double>& b2){
	TxVector2D<double> a = a2 - a1;
	TxVector2D<double> b = b2 - b1;
	TxVector2D<double> c = b1 - a1;
	//double t = (b1 - a1).Cross2(b) / (a.Cross2(b) );
	double t = (c[0] * b[1] - c[1] * b[0]) / (a[0] * b[1] - a[1] * b[0]);
	TxVector2D<double> rst = a1 + a * t;
	return rst;
}

