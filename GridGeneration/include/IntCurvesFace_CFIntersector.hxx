#ifndef _IntCurvesFace_CFIntersector_HeaderFile
#define _IntCurvesFace_CFIntersector_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif
#ifndef _BRepTopAdaptor_TopolTool_HeaderFile
#include <BRepTopAdaptor_TopolTool.hxx>
#endif
#ifndef _BRepAdaptor_HSurface_HeaderFile
#include <BRepAdaptor_HSurface.hxx>
#endif

#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif

#ifndef _IntCurveSurface_SequenceOfPnt_HeaderFile
#include <IntCurveSurface_SequenceOfPnt.hxx>
#endif

#ifndef _TColStd_SequenceOfInteger_HeaderFile
#include <TColStd_SequenceOfInteger.hxx>
#endif
#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif
#ifndef _Standard_Integer_HeaderFile
#include <Standard_Integer.hxx>
#endif

#ifndef _TopoDS_Face_HeaderFile
#include <TopoDS_Face.hxx>
#endif
#ifndef _Standard_Address_HeaderFile
#include <Standard_Address.hxx>
#endif
#ifndef _Adaptor3d_HCurve_HeaderFile
#include <Adaptor3d_HCurve.hxx>
#endif
#ifndef _GeomAbs_SurfaceType_HeaderFile
#include <GeomAbs_SurfaceType.hxx>
#endif

#ifndef _IntCurveSurface_IntersectionPoint_HeaderFile
#include <IntCurveSurface_IntersectionPoint.hxx>
#endif

#ifndef _IntCurveSurface_TransitionOnCurve_HeaderFile
#include <IntCurveSurface_TransitionOnCurve.hxx>
#endif

#ifndef _TopAbs_State_HeaderFile
#include <TopAbs_State.hxx>
#endif

class BRepTopAdaptor_TopolTool;
class BRepAdaptor_HSurface;
class TopoDS_Face;
class gp_Lin;
class Adaptor3d_HCurve;
class gp_Pnt;
class IntCurveSurface_HInter;
class gp_Pnt2d;
class Bnd_Box;



class IntCurvesFace_CFIntersector  {
public:

  void* operator new(size_t,void* anAddress) 
  {
    return anAddress;
  }
  void* operator new(size_t size) 
  {
    return Standard::Allocate(size); 
  }
  void  operator delete(void *anAddress) 
  {
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }


  Standard_EXPORT   IntCurvesFace_CFIntersector();

  //! Load a Face. <br>
  //! <br>
  //!          The Tolerance <Tol> is used to determine if the <br>
  //!          first point of the segment is near the face. In <br>
  //!          that case, the parameter of the intersection point <br>
  //!          on the line can be a negative value (greater than -Tol). <br>
  Standard_EXPORT  void Load(const TopoDS_Face& F, const Standard_Real aTol);

  //! Perform the intersection between the <br>
  //!          segment L and the loaded face. <br>
  //! <br>
  //!          PInf is the smallest parameter on the line <br>
  //!          PSup is the highest  parmaeter on the line <br>
  //! <br>
  //!          For an infinite line PInf and PSup can be <br>
  //!          +/- RealLast. <br>
  Standard_EXPORT     void Perform(const gp_Lin& L,const Standard_Real PInf,const Standard_Real PSup) ;
  //! same method for a HCurve from Adaptor3d. <br>
  //!           PInf an PSup can also be - and + INF. <br>
  Standard_EXPORT     void Perform(const Handle(Adaptor3d_HCurve)& HCu,const Standard_Real PInf,const Standard_Real PSup) ;
  //! Return the surface type <br>
  Standard_EXPORT     GeomAbs_SurfaceType SurfaceType() const;
  //! True is returned when the intersection have been computed. <br>
  Standard_Boolean IsDone() const;
  
  Standard_Integer NbPnt() const;
  //! Returns the U parameter of the ith intersection point <br>
  //!          on the surface. <br>
  Standard_Real UParameter(const Standard_Integer I) const;
  //! Returns the V parameter of the ith intersection point <br>
  //!          on the surface. <br>
  Standard_Real VParameter(const Standard_Integer I) const;
  //! Returns the parameter of the ith intersection point <br>
  //!          on the line. <br>
  Standard_Real WParameter(const Standard_Integer I) const;
  //! Returns the geometric point of the ith intersection <br>
  //!          between the line and the surface. <br>
  const gp_Pnt& Pnt(const Standard_Integer I) const;
  //! Returns the ith transition of the line on the surface. <br>
  IntCurveSurface_TransitionOnCurve Transition(const Standard_Integer I) const;
  //! Returns the ith state of the point on the face. <br>
  //!          The values can be either TopAbs_IN <br>
  //!             ( the point is in the face) <br>
  //!           or TopAbs_ON <br>
  //!             ( the point is on a boudary of the face). <br>
  TopAbs_State State(const Standard_Integer I) const;
  //! Returns the significant face used to determine <br>
  //!          the intersection. <br>
  //! <br>
  const TopoDS_Face& Face() const;
  
  Standard_EXPORT     TopAbs_State ClassifyUVPoint(const gp_Pnt2d& Puv) const;
  
  Standard_EXPORT     void Destroy() ;
  ~IntCurvesFace_CFIntersector()
  {
    Destroy();
  }
  

protected:


private:
  Standard_EXPORT     void InternalCall(const IntCurveSurface_HInter& HICS,const Standard_Real pinf,const Standard_Real psup) ;

private:
  Handle(BRepTopAdaptor_TopolTool) myTopolTool;
  Handle(BRepAdaptor_HSurface) Hsurface;
  Standard_Real Tol;
  IntCurveSurface_SequenceOfPnt SeqPnt;
  TColStd_SequenceOfInteger mySeqState;
  Standard_Boolean done;
  Standard_Integer nbpnt;
  TopoDS_Face face;
};



//============================================================================
inline Standard_Boolean IntCurvesFace_CFIntersector::IsDone() const { 
  return(done);
}
//============================================================================
inline Standard_Integer IntCurvesFace_CFIntersector::NbPnt() const { 
  return(nbpnt);
}
//============================================================================
inline const gp_Pnt& IntCurvesFace_CFIntersector::Pnt(const Standard_Integer i ) const { 
  return(SeqPnt.Value(i).Pnt()); 
}
//============================================================================
inline Standard_Real IntCurvesFace_CFIntersector::UParameter(const Standard_Integer i) const { 
  return(SeqPnt.Value(i).U()); 
}
//============================================================================
inline Standard_Real IntCurvesFace_CFIntersector::VParameter(const Standard_Integer i) const { 
  return(SeqPnt.Value(i).V()); 
}
//============================================================================
inline Standard_Real IntCurvesFace_CFIntersector::WParameter(const Standard_Integer i) const { 
  return(SeqPnt.Value(i).W()); 
}
//============================================================================
inline IntCurveSurface_TransitionOnCurve IntCurvesFace_CFIntersector::Transition(const Standard_Integer i) const { 
  return(SeqPnt.Value(i).Transition()); 
}
//============================================================================
inline TopAbs_State IntCurvesFace_CFIntersector::State(const Standard_Integer i) const { 
  return (mySeqState.Value(i) == 0) ? TopAbs_IN : TopAbs_ON;
}
//============================================================================
inline const TopoDS_Face&  IntCurvesFace_CFIntersector::Face() const { 
  return(face);
}
//============================================================================





// other Inline functions and methods (like "C++: function call" methods)


#endif
