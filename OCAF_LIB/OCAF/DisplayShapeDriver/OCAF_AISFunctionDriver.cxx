
#include "OCAF_AISFunctionDriver.ixx"
#include <OCAF_AISShape.hxx>

#include <OCAF_Object.hxx>
#include <OCAF_IFunction.hxx>
#include <OCAF_AISShape.hxx>

#include <OCAF_ColorMap.hxx>

#include <TDF_Label.hxx>

#include <TNaming_Tool.hxx>
#include <TNaming_NamedShape.hxx>

#include <AIS_InteractiveContext.hxx>
#include <AIS_Shape.hxx>
#include <AIS.hxx>

#include <BRep_Tool.hxx>

#include <TopLoc_Location.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS.hxx>

#include <TFunction_Function.hxx>
#include <TDataStd_TreeNode.hxx>
#include <TDataStd_ChildNodeIterator.hxx>

#include <Tags.hxx>
#include <CAGDDefine.hxx>
//=======================================================================
//function : GetID
//purpose  :
//=======================================================================
const Standard_GUID& OCAF_AISFunctionDriver::GetID()
{
  static Standard_GUID anID("22D22E97-C69A-11d4-8F1A-0060B0EE18E8");
  return anID;
}  


//=======================================================================
//function : OCAF_AISFunctionDriver
//purpose  : Constructor
//=======================================================================
OCAF_AISFunctionDriver::OCAF_AISFunctionDriver(){}



Standard_Boolean OCAF_AISFunctionDriver::Update (const TDF_Label& theLabel, Handle(AIS_InteractiveObject)& theAISObject) 
{
  Handle(TDataStd_TreeNode) aNode;
  Handle(TFunction_Function) aFunction;	

  if( !theLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode) )  return Standard_False;   

  OCAF_Object anInterface(aNode);

  TopoDS_Shape S = anInterface.GetObjectValue();

  Standard_Boolean IsOk = Standard_False;

  Standard_Integer theMaterialType = anInterface.GetObjResultMaterial(); 
  Quantity_NameOfColor theShadingColor = OCAF_ColorMap::getColor(theMaterialType, IsOk);


  if(S.IsNull()) return Standard_False;  

  Handle(OCAF_AISShape) anAISShape; 
  if (theAISObject.IsNull()){
    anAISShape = new OCAF_AISShape(S); 
  }else{ 
    anAISShape = Handle(OCAF_AISShape)::DownCast(theAISObject); 
    if (anAISShape.IsNull()) { 
      anAISShape = new OCAF_AISShape(S); 
    }else{ 
      TopoDS_Shape oldShape = anAISShape->Shape(); 
      if(oldShape != S) { 
	anAISShape->ResetTransformation();
	anAISShape->Set(S);
	//anAISShape->UpdateSelection(); 
	//anAISShape->SetToUpdate();
      }
    } 
    //anAISShape->SetInfiniteState(S.Infinite()); 
  }



  anAISShape->SetInfiniteState(S.Infinite()); 
  anAISShape->UpdateSelection(); 
  anAISShape->SetToUpdate();


  anAISShape->SetShadingColor(theShadingColor);

  /*
  if(!IsOk){
    cout<<"error----------------------------------OCAF_AISFunctionDriver::Update"<<endl;
  }
  //*/

  theAISObject = anAISShape; 

  return Standard_True; 
}

