
#ifndef _OCAF_IBox_HeaderFile
#define _OCAF_IBox_HeaderFile

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


#include <Tags.hxx>


class OCAF_IBox : public OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress)  {  return anAddress;  }
  inline void* operator new(size_t size)  {  return Standard::Allocate(size); }
  inline void  operator delete(void *anAddress)  {   if (anAddress) Standard::Free((Standard_Address&)anAddress);  }
  
  // Methods PUBLIC
  // 
  Standard_EXPORT static const Standard_GUID& GetID() ;

  
  Standard_EXPORT static  Handle(TDataStd_TreeNode) MakeBox_FunctionNode( const Handle(TDataStd_TreeNode)& anObjectNode,
									  Standard_Integer theType,
									  TCollection_ExtendedString& theError);
  

  //Standard_EXPORT static  Standard_Boolean MakeBox_Excute( const Handle(TDataStd_TreeNode)& aFunctionNode, TCollection_ExtendedString& theError);
  
  Standard_EXPORT   Standard_Boolean MakeBox_Execute(  TCollection_ExtendedString& theError);

  Standard_EXPORT   OCAF_IBox(const Handle(TDataStd_TreeNode)& aTreeNode);

  Standard_EXPORT   void   SetX(double theX) { SetReal(BOX_ARG_X, theX); }
  Standard_EXPORT   double GetX() { return GetReal(BOX_ARG_X); }

  Standard_EXPORT   void   SetY(double theY) { SetReal(BOX_ARG_Y, theY); }
  Standard_EXPORT   double GetY() { return GetReal(BOX_ARG_Y); }

  Standard_EXPORT   void   SetZ(double theZ) { SetReal(BOX_ARG_Z, theZ); }
  Standard_EXPORT   double GetZ() { return GetReal(BOX_ARG_Z); }


  Standard_EXPORT   void   SetDX(double theX) { SetReal(BOX_ARG_DX, theX); }
  Standard_EXPORT   double GetDX() { return GetReal(BOX_ARG_DX); }

  Standard_EXPORT   void   SetDY(double theY) { SetReal(BOX_ARG_DY, theY); }
  Standard_EXPORT   double GetDY() { return GetReal(BOX_ARG_DY); }

  Standard_EXPORT   void   SetDZ(double theZ) { SetReal(BOX_ARG_DZ, theZ); }
  Standard_EXPORT   double GetDZ() { return GetReal(BOX_ARG_DZ); }


  Standard_EXPORT   void                        SetPoint1( const Handle(TDataStd_TreeNode)& theRefPoint1) { SetReference(BOX_ARG_PNT1, theRefPoint1); }
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint1Node() { return GetReferenceNode(BOX_ARG_PNT1); }
  Standard_EXPORT   TopoDS_Shape                GetPoint1() { return GetReference(BOX_ARG_PNT1); }

  Standard_EXPORT   void                        SetPoint2( const Handle(TDataStd_TreeNode)& theRefPoint2) { SetReference(BOX_ARG_PNT2, theRefPoint2); }
  Standard_EXPORT   Handle(TDataStd_TreeNode)   GetPoint2Node() { return GetReferenceNode(BOX_ARG_PNT2); }
  Standard_EXPORT   TopoDS_Shape                GetPoint2() { return GetReference(BOX_ARG_PNT2); }

  Standard_EXPORT   ~OCAF_IBox() {}


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
