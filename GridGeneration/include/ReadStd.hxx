#ifndef _ReadStd_HeaderFile
#define _ReadStd_HeaderFile


#ifndef _Model_Ctrl_HeaderFile
#include <Model_Ctrl.hxx>
#endif


Standard_Boolean BeginReadOCCStd();

//Standard_Boolean ReadMarkedShape(const Standard_CString& SPath, Model_Ctrl*& theModelCtrl);

Standard_Boolean ReadOCCStd(const Standard_CString& SPath, Model_Ctrl*& theModelCtrl);

Standard_Boolean EndReadOCCStd();

#endif
