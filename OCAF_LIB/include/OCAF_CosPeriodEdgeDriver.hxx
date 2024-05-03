//                     Copyright (C) 2014  by
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

#ifndef _OCAF_CosPeriodEdgeDriver_HeaderFile
#define _OCAF_CosPeriodEdgeDriver_HeaderFile

#ifndef _OCAF_Driver_HeaderFile
#include <OCAF_Driver.hxx>
#endif

#ifndef _Geom_BSplineCurve_HeaderFile
#include <Geom_BSplineCurve.hxx>
#endif

DEFINE_STANDARD_HANDLE(OCAF_CosPeriodEdgeDriver,OCAF_Driver)

class OCAF_CosPeriodEdgeDriver : public OCAF_Driver 
{
public:

  // Methods PUBLIC
  // 
  Standard_EXPORT OCAF_CosPeriodEdgeDriver();
  

  Standard_EXPORT Handle(Geom_BSplineCurve) InterpolateOnePeriod(const Handle(TDataStd_TreeNode)& aNode,
								 const Standard_Integer thePeriodIndx) const;

  Standard_EXPORT Handle(Geom_BSplineCurve) InterpolateOnePeriod(const Handle(TDataStd_TreeNode)& aNode,
								 const Standard_Integer theSectionNum,
								 const Standard_Integer theSectionIndx,
								 const Standard_Integer thePeriodIndx) const;

  Standard_EXPORT TopoDS_Shape Interpolate_0(const Handle(TDataStd_TreeNode)& aNode) const;
  Standard_EXPORT TopoDS_Shape Interpolate_1(const Handle(TDataStd_TreeNode)& aNode) const;
  Standard_EXPORT TopoDS_Shape Interpolate_2(const Handle(TDataStd_TreeNode)& aNode) const;

  Standard_EXPORT TopoDS_Shape Polygon_0(const Handle(TDataStd_TreeNode)& aNode) const;
  Standard_EXPORT TopoDS_Shape Polygon_1(const Handle(TDataStd_TreeNode)& aNode) const;

  Standard_EXPORT virtual  Standard_Integer Execute(Handle(TFunction_Logbook)& theLogbook) const;


  //DynamicType

  DEFINE_STANDARD_RTTIEXT(OCAF_CosPeriodEdgeDriver,OCAF_Driver)

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
