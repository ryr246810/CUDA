//                     Copyright (C) 2010 by
//  
//                      Wang Yue, China
//  
// This software is furnished in accordance with the terms and conditions
// of the contract and with the inclusion of the above copyright notice.
// This software or any other copy thereof may not be provided or otherwise
// be made available to any other person. No title to an ownership of the
// software is hereby transferred.
//  
// At the termination of the contract, the software and all copies of this
// software must be deleted.
//
#ifndef _OCAF_ICurve_HeaderFile
#define _OCAF_ICurve_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _Standard_GUID_HeaderFile
#include <Standard_GUID.hxx>
#endif

#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif

#ifndef _TCollection_ExtendedString_HeaderFile
#include <TCollection_ExtendedString.hxx>
#endif

#ifndef _TDataStd_Real_HeaderFile
#include <TDataStd_Real.hxx>
#endif

#ifndef _OCAF_IFunction_HeaderFile
#include <OCAF_IFunction.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif


#include <TColStd_HArray1OfReal.hxx>

#include <Tags.hxx>



class OCAF_ICurve : public OCAF_IFunction 
{
public:
  inline void* operator new(size_t,void* anAddress)  {  return anAddress; }
  inline void* operator new(size_t size)  { return Standard::Allocate(size);  }
  inline void  operator delete(void *anAddress)  {  if (anAddress) Standard::Free((Standard_Address&)anAddress); }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeCurve_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									    Standard_Integer theType,
									    TCollection_ExtendedString& theError);

  Standard_EXPORT   Standard_Boolean MakeCurve_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_ICurve(const Handle(TDataStd_TreeNode)& aTreeNode);


  Standard_EXPORT   void	SetRowCount(int theParam) { SetInteger(CURVE_ROWCOUNT, theParam); }
  Standard_EXPORT   int		GetRowCount() { return GetInteger(CURVE_ROWCOUNT); }


  Standard_EXPORT	void	SetArray (const Handle(TColStd_HArray1OfReal)& theArray){ SetRealArray (CURVE_ARRAY,theArray);}
  Standard_EXPORT	Handle(TColStd_HArray1OfReal) GetArray(){ return GetRealArray(CURVE_ARRAY);}

  Standard_EXPORT   ~OCAF_ICurve() {};

protected:

 // Methods PROTECTED
 // 


 // Fields PROTECTED
 //


private: 

 // Methods PRIVATE
 // 


 // Fields PRIVATE
 //


};


// other inline functions and methods (like "C++: function call" methods)
//


#endif
