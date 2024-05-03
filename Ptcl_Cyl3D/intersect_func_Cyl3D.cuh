#ifndef INTERSECT_Cyl3D_FUNC
#define INTERSECT_Cyl3D_FUNC

#include <TxVector2D.h>


bool detect_intersect_Cyl3D(const TxVector2D<double>& a,  const TxVector2D<double>& b, const TxVector2D<double>& c,  const TxVector2D<double>& d);

TxVector2D<double> get_intersect_pnt_Cyl3D(const TxVector2D<double>& a,  const TxVector2D<double>& b, const TxVector2D<double>& c,  const TxVector2D<double>& d);



#endif	
