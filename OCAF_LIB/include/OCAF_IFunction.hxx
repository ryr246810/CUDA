
#ifndef _OCAF_IFunction_HeaderFile
#define _OCAF_IFunction_HeaderFile

#ifndef _OCAF_ObjectType_HeaderFile
#include <OCAF_ObjectType.hxx>
#endif

#ifndef _TDF_Label_HeaderFile
#include <TDF_Label.hxx>
#endif

#ifndef _TopoDS_Shape_HeaderFile
#include <TopoDS_Shape.hxx>
#endif

#ifndef _TCollection_ExtendedString_HeaderFile
#include <TCollection_ExtendedString.hxx>
#endif

#ifndef _TPrsStd_AISPresentation_HeaderFile
#include <TPrsStd_AISPresentation.hxx>
#endif

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

#ifndef _TDF_AttributeMap_HeaderFile
#include <TDF_AttributeMap.hxx>
#endif

#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif

#ifndef _MMgt_TShared_HeaderFile
#include <MMgt_TShared.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif

#ifndef _gp_Pnt_HeaderFile
#include <gp_Pnt.hxx>
#endif

#ifndef _TDF_LabelMap_HeaderFile
#include <TDF_LabelMap.hxx>
#endif

#ifndef _TDF_LabelList_HeaderFile
#include <TDF_LabelList.hxx>
#endif

#ifndef _TDF_AttributeList_HeaderFile
#include <TDF_AttributeList.hxx>
#endif

#ifndef _TDF_LabelSequence_HeaderFile
#include <TDF_LabelSequence.hxx>
#endif

#ifndef _TDF_AttributeSequence_HeaderFile
#include <TDF_AttributeSequence.hxx>
#endif

#ifndef _TNaming_NamedShape_HeaderFile
#include <TNaming_NamedShape.hxx>
#endif

#ifndef _TColStd_HArray1OfReal_HeaderFile
#include <TColStd_HArray1OfReal.hxx>
#endif

class TFunction_Logbook;

class OCAF_IFunction {

public:

  inline void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  inline void* operator new(size_t size) 
  { 
    return Standard::Allocate(size); 
  }
  inline void  operator delete(void *anAddress) 
  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  
  Standard_EXPORT   OCAF_IFunction(const Handle(TDataStd_TreeNode)& aTreeNode); //ok

  Standard_EXPORT static OCAF_ObjectType GetObjectType(const Handle(TDataStd_TreeNode)& Object);  //ok

  Standard_EXPORT static void AddLabels(const Handle(TDataStd_TreeNode)& aFunctionNode, 
					Handle(TFunction_Logbook)& theLog); // ok

  Standard_EXPORT static void AddLogBooks(const Handle(TDataStd_TreeNode)& theFunctionNode, 
					  Handle(TFunction_Logbook)& theLog);

  Standard_EXPORT   void SetName(const TCollection_ExtendedString& aName); // ok
  Standard_EXPORT   Standard_Boolean HasName(); //ok
  Standard_EXPORT   TCollection_ExtendedString GetName() const; //ok

  // Methods PUBLIC
  // 
  Standard_EXPORT TopoDS_Shape GetFunctionResult(); // ok


  Standard_EXPORT   Standard_Boolean SetType(Standard_Integer theType);  // ok
  Standard_EXPORT   Standard_Integer GetType() const;  // ok

  Standard_EXPORT   void SetCenter(const Standard_Real X,
				   const Standard_Real Y,
				   const Standard_Real Z) ;  // ok
  Standard_EXPORT   Standard_Boolean HasCenter() ; // ok

  Standard_EXPORT   void GetCenter(Standard_Real& X,
				   Standard_Real& Y,
				   Standard_Real& Z) ; // ok
  Standard_EXPORT   gp_Pnt GetCenterPnt();   // ok


  // some methods
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Standard_EXPORT   TDF_Label GetEntry() const;

  Standard_EXPORT   Standard_Boolean SetOrientation(Standard_Integer theType);  // ok
  Standard_EXPORT   Standard_Integer GetOrientation() const;  // ok

  Standard_EXPORT   void             SetReal(int thePosition, 
					     double theValue);  // ok
  Standard_EXPORT   double           GetReal(int thePosition);  // ok

  Standard_EXPORT   void             SetInteger(int thePosition, 
						Standard_Integer theValue);  // ok
  Standard_EXPORT   Standard_Integer GetInteger(int thePosition);  // ok

  Standard_EXPORT   void SetRealArray (int thePosition, 
				       const Handle(TColStd_HArray1OfReal)& theArray);  // ok
  Standard_EXPORT   Handle(TColStd_HArray1OfReal) GetRealArray(int thePosition);  // ok

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



  // of a ARGUMENTTAG
  // OCAF_IFunction_1.cxx
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Standard_EXPORT   Standard_Boolean          SetReference(int thePosition, 
							   const Handle(TDataStd_TreeNode)& theReference); //ok
  Standard_EXPORT   TopoDS_Shape              GetReference(int thePosition);  //ok
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetReferenceNode(int thePosition);

  Standard_EXPORT   Standard_Boolean          SetFuncReference(int thePosition, 
							       const Handle(TDataStd_TreeNode)& theReference); //ok
  Standard_EXPORT   TopoDS_Shape              GetFuncReference(int thePosition);  // ok
  Standard_EXPORT   Handle(TDataStd_TreeNode) GetFuncReferenceNode(int thePosition); //ok
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



  // of a ChildTag of ARGUMENTTAG
  // OCAF_IFunction_2.cxx
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Standard_EXPORT   Standard_Boolean          SetReferenceOfTag(int theTag, 
								int theChildTag, 
								const Handle(TDataStd_TreeNode)& theReference);  //ok

  Standard_EXPORT   Handle(TDataStd_TreeNode) GetReferenceNodeOfTag(int theTag, 
								    int theChildTag); // ok

  Standard_EXPORT   TopoDS_Shape              GetReferenceOfTag(int theTag, 
								int theChildTag);  //ok


  Standard_EXPORT   void                      GetArgumentsMapOfTag(const int theTag, 
								   TDF_AttributeMap& anAttributeMap);  // ok
  Standard_EXPORT   void                      GetArgumentsSequenceOfTag(const int theTag,
									TDF_AttributeSequence& anAttributeSequence); //ok

  Standard_EXPORT   void                      ClearAllArgumentsOfTag(int aTag);  //ok
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




  Standard_EXPORT ~OCAF_IFunction() {};

protected:

  // Methods PROTECTED
  // 

  // Fields PROTECTED
  //
  Handle(TDataStd_TreeNode) myTreeNode;


private: 
  
  // Methods PRIVATE
  // 
  
  
  // Fields PRIVATE
  //
};





// other inline functions and methods (like "C++: function call" methods)
//


#endif
