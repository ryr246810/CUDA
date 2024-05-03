

#ifndef _IntCurvesFace_NewIntersector_HeaderFile
#include <IntCurvesFace_NewIntersector.hxx>
#endif

#ifndef _BRepTopAdaptor_TopolTool_HeaderFile
#include <BRepTopAdaptor_TopolTool.hxx>
#endif
#ifndef _BRepAdaptor_HSurface_HeaderFile
#include <BRepAdaptor_HSurface.hxx>
#endif
#ifndef _TopoDS_Face_HeaderFile
#include <TopoDS_Face.hxx>
#endif
#ifndef _gp_Lin_HeaderFile
#include <gp_Lin.hxx>
#endif
#ifndef _Adaptor3d_HCurve_HeaderFile
#include <Adaptor3d_HCurve.hxx>
#endif
#ifndef _gp_Pnt_HeaderFile
#include <gp_Pnt.hxx>
#endif
#ifndef _IntCurveSurface_HInter_HeaderFile
#include <IntCurveSurface_HInter.hxx>
#endif
#ifndef _gp_Pnt2d_HeaderFile
#include <gp_Pnt2d.hxx>
#endif
#ifndef _Bnd_Box_HeaderFile
#include <Bnd_Box.hxx>
#endif

#include <IntCurveSurface_ThePolyhedronToolOfHInter.hxx>
#include <Bnd_BoundSortBox.hxx>

#include <IntCurveSurface_IntersectionPoint.hxx>
#include <gp_Lin.hxx>
#include <TopoDS_Face.hxx>
#include <TopAbs.hxx>


#include <IntCurveSurface_HInter.hxx>
#include <BRepAdaptor_HSurface.hxx>
#include <Geom_Line.hxx>
#include <gp_Pnt2d.hxx>
#include <BRepClass_FaceClassifier.hxx>





#include <Adaptor3d_HSurfaceTool.hxx>
#include <Adaptor3d_HCurve.hxx>
#include <Bnd_Box.hxx>
#include <Intf_Tool.hxx>
#include <IntCurveSurface_ThePolyhedronOfHInter.hxx>
#include <IntCurveSurface_ThePolygonOfHInter.hxx>
#include <IntCurveSurface_SequenceOfPnt.hxx>


#include <GeomAdaptor_Curve.hxx>

#include <GeomAdaptor_HCurve.hxx>

#include <BRepAdaptor_HSurface.hxx>

GeomAbs_SurfaceType IntCurvesFace_NewIntersector::SurfaceType() const { 
  return(Adaptor3d_HSurfaceTool::GetType(Hsurface));
}
 
 
//============================================================================
IntCurvesFace_NewIntersector::IntCurvesFace_NewIntersector(const TopoDS_Face& Face,const Standard_Real aTol)
  : Tol(aTol),done(Standard_False),nbpnt(0)
{ 
  BRepAdaptor_Surface               surface;
  face = Face;
  surface.Initialize(Face,Standard_True);
  Hsurface = new BRepAdaptor_HSurface(surface);
  myTopolTool = new BRepTopAdaptor_TopolTool(Hsurface);
}


//============================================================================

void IntCurvesFace_NewIntersector::InternalCall(const IntCurveSurface_HInter &HICS, const Standard_Real parinf, const Standard_Real parsup) {
  if(HICS.IsDone()) {
    for(Standard_Integer index=HICS.NbPoints(); index>=1; index--) {  
      const IntCurveSurface_IntersectionPoint& HICSPointindex = HICS.Point(index);
      gp_Pnt2d Puv(HICSPointindex.U(),HICSPointindex.V());
      
      TopAbs_State currentstate = ClassifyUVPoint(Puv);

      if(currentstate==TopAbs_IN || currentstate==TopAbs_ON) { 
	Standard_Real HICSW = HICSPointindex.W();
	if(HICSW >= parinf && HICSW <= parsup ) { 
	  Standard_Real U          = HICSPointindex.U();
	  Standard_Real V          = HICSPointindex.V();
	  Standard_Real W          = HICSW; 
	  IntCurveSurface_TransitionOnCurve transition = HICSPointindex.Transition();
	  gp_Pnt pnt        = HICSPointindex.Pnt();
	  //Standard_Integer anIntState = (currentstate == TopAbs_IN || currentstate == TopAbs_ON) ? 0 : 1;
	  Standard_Integer anIntState = (currentstate == TopAbs_IN) ? 0 : 1;

	  if(transition != IntCurveSurface_Tangent && face.Orientation()==TopAbs_REVERSED) { 
	    if(transition == IntCurveSurface_In) 
	      transition = IntCurveSurface_Out;
	    else 
	      transition = IntCurveSurface_In;
	  }
	  //----- Insertion du point 
	  if(nbpnt==0) { 
	    IntCurveSurface_IntersectionPoint PPP(pnt,U,V,W,transition);
	    SeqPnt.Append(PPP);
	    mySeqState.Append(anIntState);
	  }
	  else { 
	    Standard_Integer i = 1;
	    Standard_Integer b = nbpnt+1;                    
	    while(i<=nbpnt) {
	      const IntCurveSurface_IntersectionPoint& Pnti=SeqPnt.Value(i);
	      Standard_Real wi = Pnti.W();
	      if(wi >= W) { b=i; i=nbpnt; }
	      i++;
	    }
	    IntCurveSurface_IntersectionPoint PPP(pnt,U,V,W,transition);
// 	    if(b>nbpnt)          { SeqPnt.Append(PPP); } 
// 	    else if(b>0)         { SeqPnt.InsertBefore(b,PPP); } 
	    if(b>nbpnt) {
	      SeqPnt.Append(PPP);
	      mySeqState.Append(anIntState);
	    } else if(b>0) {
	      SeqPnt.InsertBefore(b,PPP);
	      mySeqState.InsertBefore(b, anIntState);
	    }
	  }
	  nbpnt++;
	} 
      } //-- classifier state is IN or ON
    } //-- Loop on Intersection points.
  } //-- HICS.IsDone()
}

//--------------------------------------------------------------------------------


void IntCurvesFace_NewIntersector::Perform(const gp_Lin& L,const Standard_Real ParMin,const Standard_Real ParMax) { 
  done = Standard_True;
  SeqPnt.Clear();
  mySeqState.Clear();
  nbpnt = 0;
  IntCurveSurface_HInter            HICS; 
  
  Handle(Geom_Line) geomline = new Geom_Line(L);
  GeomAdaptor_Curve LL(geomline);
  
  //--
  Handle(GeomAdaptor_HCurve) HLL  = new GeomAdaptor_HCurve(LL);
  //-- 
  Standard_Real parinf=ParMin;
  Standard_Real parsup=ParMax;

  HICS.Perform(HLL,Hsurface);
  InternalCall(HICS,parinf,parsup);
}



//============================================================================
void IntCurvesFace_NewIntersector::Perform(const Handle(Adaptor3d_HCurve)& HCu,const Standard_Real ParMin,const Standard_Real ParMax) { 
  done = Standard_True;
  SeqPnt.Clear();
  mySeqState.Clear();
  nbpnt = 0;
  IntCurveSurface_HInter            HICS; 
  
  //-- 
  Standard_Real parinf=ParMin;
  Standard_Real parsup=ParMax;

  HICS.Perform(HCu,Hsurface);
  InternalCall(HICS,parinf,parsup);
}

TopAbs_State IntCurvesFace_NewIntersector::ClassifyUVPoint(const gp_Pnt2d& Puv) const { 
  TopAbs_State state = myTopolTool->Classify(Puv,Tol);
  return(state);
}

//============================================================================
void IntCurvesFace_NewIntersector::Destroy() { 
}
