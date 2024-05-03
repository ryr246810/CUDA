// OCCT Includes
#include <Prs3d_Presentation.hxx>
#include <PrsMgr_PresentationManager3d.hxx>


//#include <AIS_Drawer.hxx>
#include <AIS_InteractiveContext.hxx>
#include <Graphic3d_AspectFillArea3d.hxx>
#include <Prs3d_ShadingAspect.hxx>
#include <SelectBasics_SensitiveEntity.hxx>
#include <SelectMgr_EntityOwner.hxx>
#include <StdSelect_BRepOwner.hxx>
#include <SelectMgr_IndexedMapOfOwner.hxx>
#include <SelectMgr_Selection.hxx>
#include <StdSelect_DisplayMode.hxx>
#include <StdPrs_WFShape.hxx>
#include <TColStd_IndexedMapOfInteger.hxx>
#include <TColStd_ListIteratorOfListOfInteger.hxx>
#include <TColStd_ListOfInteger.hxx>
#include <TopExp.hxx>
#include <TopoDS_Shape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

#include <BRep_Tool.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>
#include <gp_Vec.hxx>
#include <Prs3d_Arrow.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GCPnts_AbscissaPoint.hxx>

//#include <StdPrs_WFDeflectionShape.hxx>
#include <Prs3d_LineAspect.hxx>
#include <OpenGl_Group.hxx>


#include "OCAF_AISShape.ixx"


typedef enum {
  Wireframe        = AIS_WireFrame,       //!< wireframe
  Shading          = AIS_Shaded,          //!< shadin
  ShadingWithEdges,                       //!< shading with edges
  TexturedShape,                          //!< texture
  CustomHighlight                         //!< fields
} DispMode;


//=======================================================================
//function : OCAF_AISShape
//purpose  : Constructor
//=======================================================================
OCAF_AISShape::
OCAF_AISShape (const TopoDS_Shape& theShape)
  : AIS_Shape(theShape)
{
  m_ShadingColor = Quantity_Color( Quantity_NOC_GOLDENROD );
}


//=======================================================================
//function : 
//purpose  : 
//=======================================================================
void 
OCAF_AISShape::
SetShadingColor(Quantity_NameOfColor aName)
{
  m_ShadingColor = Quantity_Color(aName);
}


//=======================================================================
//function : Compute
//purpose  : Compute a presentation
//=======================================================================
void 
OCAF_AISShape::
Compute (const Handle(PrsMgr_PresentationManager3d)& aPresentationManager,
	 const Handle(Prs3d_Presentation)& aPrs,
	 const Standard_Integer aMode)
{
  if (IsInfinite()) aPrs->SetInfiniteState(Standard_True); 

  switch (aMode) 
    {
    case Wireframe:
    case CustomHighlight:
      {
	SetColor(m_ShadingColor);
	Handle(Prs3d_LineAspect) anAspect = Attributes()->WireAspect();
	anAspect->SetColor( m_ShadingColor );
	Attributes()->SetWireAspect( anAspect );

	StdPrs_WFShape::Add(aPrs,myshape,myDrawer);
	break;
      }
    case StdSelect_DM_Shading://Shading:
      {
	shadingMode(aPresentationManager, aPrs, aMode);
	break;
      }
      
    case ShadingWithEdges:
      {
	myDrawer->SetFaceBoundaryDraw( Standard_True );
	shadingMode(aPresentationManager, aPrs, Shading);
	break;
      }
    case TexturedShape:
      {
	AIS_Shape::Compute(aPresentationManager, aPrs, aMode);
      }
    }


  if (isShowVectors()) {
    TopExp_Explorer Exp ( myshape, TopAbs_EDGE );
    for ( ; Exp.More(); Exp.Next() ) {
      TopoDS_Vertex aV1, aV2;
      TopoDS_Edge anEdgeE = TopoDS::Edge(Exp.Current());

      if ( anEdgeE.IsNull() ) continue;

      TopExp::Vertices(anEdgeE, aV1, aV2);
      gp_Pnt aP1 = BRep_Tool::Pnt(aV1);
      gp_Pnt aP2 = BRep_Tool::Pnt(aV2);

      double fp,lp;
      gp_Vec aDirVec;
      Handle(Geom_Curve) C = BRep_Tool::Curve(anEdgeE,fp,lp);

      if ( C.IsNull() ) continue;

      if ( anEdgeE.Orientation() == TopAbs_FORWARD )
        C->D1(lp, aP2, aDirVec);
      else {
        C->D1(fp, aP1, aDirVec);
        aP2 = aP1;
      }

      GeomAdaptor_Curve aAdC;
      aAdC.Load(C, fp, lp);
      Standard_Real aDist = GCPnts_AbscissaPoint::Length(aAdC, fp, lp); 
     
      if (aDist > gp::Resolution()) {
        gp_Dir aDir;
        if ( anEdgeE.Orientation() == TopAbs_FORWARD )
          aDir = aDirVec;
        else
          aDir = -aDirVec;

	//Prs3d_Arrow::Draw(aPrs, aP2, aDir, M_PI/180.*5., aDist/10.);
	Handle(OpenGl_Group) aGroup = Handle(OpenGl_Group)::DownCast(aPrs->NewGroup());
        Prs3d_Arrow::Draw(aGroup, aP2, aDir, M_PI/180.*5., aDist/10.);
      }
    }
  }
}


void 
OCAF_AISShape::
shadingMode(const Handle(PrsMgr_PresentationManager3d)& aPresentationManager,
	    const Handle(Prs3d_Presentation)& aPrs,
	    const Standard_Integer aMode)
{
  /*
  myDrawer->ShadingAspect()->Aspect()->SetDistinguishOn();
  
  Graphic3d_MaterialAspect aMatAspect;
  aMatAspect.SetTransparency(myTransparency);
  
  aMatAspect.SetAmbient( 0.5 );
  aMatAspect.SetDiffuse( 0.5 );
  aMatAspect.SetEmissive( 0.5 );
  aMatAspect.SetShininess(0.5 );
  aMatAspect.SetSpecular( 0.5 );
  
  myDrawer->ShadingAspect()->Aspect()->SetFrontMaterial(aMatAspect);
  myDrawer->ShadingAspect()->Aspect()->SetBackMaterial(Graphic3d_NOM_JADE);
  
  Graphic3d_MaterialAspect FMat = myDrawer->ShadingAspect()->Aspect()->FrontMaterial();
  Graphic3d_MaterialAspect BMat = myDrawer->ShadingAspect()->Aspect()->BackMaterial();
  
  FMat.SetTransparency(myTransparency);
  BMat.SetTransparency(myTransparency);
  
  myDrawer->ShadingAspect()->Aspect()->SetFrontMaterial(FMat);
  myDrawer->ShadingAspect()->Aspect()->SetBackMaterial(BMat);
  
  myDrawer->ShadingAspect()->SetColor(m_ShadingColor);
  
  AIS_Shape::Compute(aPresentationManager, aPrs, aMode);
  //*/

  /*
  myDrawer->ShadingAspect()->Aspect()->SetDistinguishOn();
  Graphic3d_MaterialAspect aMatAspect(Graphic3d_NOM_PLASTIC);

  aMatAspect.SetTransparency(Transparency());
  Graphic3d_MaterialAspect currentFrontMaterial = myDrawer->ShadingAspect()->Aspect()->FrontMaterial();
  Graphic3d_MaterialAspect currentBackMaterial  = myDrawer->ShadingAspect()->Aspect()->BackMaterial();
  myDrawer->ShadingAspect()->Aspect()->SetFrontMaterial( currentFrontMaterial );
  myDrawer->ShadingAspect()->Aspect()->SetBackMaterial ( currentBackMaterial  );

  if(myDrawer->ShadingAspect()->Aspect()->FrontMaterial().MaterialType( Graphic3d_MATERIAL_ASPECT ))
    myDrawer->ShadingAspect()->SetColor(m_ShadingColor);
  else
    myDrawer->ShadingAspect()->SetColor(myDrawer->ShadingAspect()->Aspect()->FrontMaterial().AmbientColor());

  AIS_Shape::Compute(aPresentationManager, aPrs, aMode);
  //*/


  myDrawer->ShadingAspect()->Aspect()->SetDistinguishOn();
  Graphic3d_MaterialAspect aMatAspect(Graphic3d_NOM_PLASTIC);

  aMatAspect.SetTransparency(Transparency());
  Graphic3d_MaterialAspect currentFrontMaterial = myDrawer->ShadingAspect()->Aspect()->FrontMaterial();
  Graphic3d_MaterialAspect currentBackMaterial  = myDrawer->ShadingAspect()->Aspect()->BackMaterial();
  myDrawer->ShadingAspect()->Aspect()->SetFrontMaterial( currentFrontMaterial );
  myDrawer->ShadingAspect()->Aspect()->SetBackMaterial ( currentBackMaterial  );

  myDrawer->ShadingAspect()->SetColor(m_ShadingColor);

  Handle(OpenGl_Group) aGroup = Handle(OpenGl_Group)::DownCast(aPrs->NewGroup());
  AIS_Shape::Compute(aPresentationManager, aPrs, aMode);
  //AIS_Shape::Compute(aPresentationManager, aGroup, aMode);
}



void OCAF_AISShape::SetDisplayVectors(bool isDisplayed)
{
  myDisplayVectors = isDisplayed;
}



bool OCAF_AISShape::isShowVectors() 
{ 
  return myDisplayVectors; 
}



void 
OCAF_AISShape::
SetTransparency(const Standard_Real aValue)
{
  if(aValue<0.0 || aValue>1.0){
    UnsetTransparency();
  };
  if(aValue<=0.05){
    UnsetTransparency();
  };

  Graphic3d_MaterialAspect FMat = myDrawer->ShadingAspect()->Aspect()->FrontMaterial();
  Graphic3d_MaterialAspect BMat = myDrawer->ShadingAspect()->Aspect()->BackMaterial();

  FMat.SetTransparency(aValue); 
  BMat.SetTransparency(aValue);

  myDrawer->ShadingAspect()->Aspect()->SetFrontMaterial(FMat);
  myDrawer->ShadingAspect()->Aspect()->SetBackMaterial(BMat);

  myDrawer->SetTransparency(aValue);
}

