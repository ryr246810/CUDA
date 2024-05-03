#ifndef _IntCurvesFace_NewShapeIntersector_HeaderFile
#define _IntCurvesFace_NewShapeIntersector_HeaderFile

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif
#ifndef _Standard_Integer_HeaderFile
#include <Standard_Integer.hxx>
#endif
#ifndef _Standard_Address_HeaderFile
#include <Standard_Address.hxx>
#endif
#ifndef _BRepTopAdaptor_SeqOfPtr_HeaderFile
#include <BRepTopAdaptor_SeqOfPtr.hxx>
#endif
#ifndef _TColStd_SequenceOfInteger_HeaderFile
#include <TColStd_SequenceOfInteger.hxx>
#endif
#ifndef _TColStd_SequenceOfReal_HeaderFile
#include <TColStd_SequenceOfReal.hxx>
#endif
#ifndef _Standard_Real_HeaderFile
#include <Standard_Real.hxx>
#endif
#ifndef _Adaptor3d_HCurve_HeaderFile
#include <Adaptor3d_HCurve.hxx>
#endif
#ifndef _IntCurveSurface_TransitionOnCurve_HeaderFile
#include <IntCurveSurface_TransitionOnCurve.hxx>
#endif
#ifndef _TopAbs_State_HeaderFile
#include <TopAbs_State.hxx>
#endif

#ifndef _IntCurvesFace_NewIntersector_HeaderFile
#include <IntCurvesFace_NewIntersector.hxx>
#endif

class TopoDS_Shape;
class gp_Lin;
class Adaptor3d_HCurve;
class gp_Pnt;
class TopoDS_Face;



class IntCurvesFace_NewShapeIntersector  {
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

  
  Standard_EXPORT   IntCurvesFace_NewShapeIntersector();
  
  Standard_EXPORT     void Load(const TopoDS_Shape& Sh,const Standard_Real Tol) ;
  //! Perform the intersection between the <br>
  //!          segment L and the loaded shape. <br>
  //! <br>
  //!          PInf is the smallest parameter on the line <br>
  //!          PSup is the highest  parammter on the line <br>
  //! <br>
  //!          For an infinite line PInf and PSup can be <br>
  //!          +/- RealLast. <br>
  Standard_EXPORT     void Perform(const gp_Lin& L,const Standard_Real PInf,const Standard_Real PSup) ;
  //! Perform the intersection between the <br>
  //!          segment L and the loaded shape. <br>
  //! <br>
  //!          PInf is the smallest parameter on the line <br>
  //!          PSup is the highest  parammter on the line <br>
  //! <br>
  //!          For an infinite line PInf and PSup can be <br>
  //!          +/- RealLast. <br>
  Standard_EXPORT     void PerformNearest(const gp_Lin& L,const Standard_Real PInf,const Standard_Real PSup) ;
  //! same method for a HCurve from Adaptor3d. <br>
  //!           PInf an PSup can also be - and + INF. <br>
  Standard_EXPORT     void Perform(const Handle(Adaptor3d_HCurve)& HCu,const Standard_Real PInf,const Standard_Real PSup) ;
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
  const TopoDS_Face& Face(const Standard_Integer I) const;
  //! Internal method. Sort the result on the Curve <br>
  //!          parameter. <br>
  Standard_EXPORT     void SortResult() ;
  
  Standard_EXPORT     void Destroy() ;
  ~IntCurvesFace_NewShapeIntersector()
  {
    Destroy();
  }
 
protected:
  
private:
  Standard_Boolean done;
  Standard_Integer nbfaces;
  Standard_Address PtrJetons;
  Standard_Address PtrJetonsIndex;
  BRepTopAdaptor_SeqOfPtr PtrIntersector;
  TColStd_SequenceOfInteger IndexPt;
  TColStd_SequenceOfInteger IndexFace;
  TColStd_SequenceOfInteger IndexIntPnt;
  TColStd_SequenceOfReal IndexPar;
};


inline Standard_Integer IntCurvesFace_NewShapeIntersector::NbPnt() const { 
  return(IndexPt.Length());
}

inline  Standard_Real IntCurvesFace_NewShapeIntersector::UParameter(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->UParameter(IndexIntPnt(IndexPt(i))));
}

inline  Standard_Real IntCurvesFace_NewShapeIntersector::VParameter(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->VParameter(IndexIntPnt(IndexPt(i))));
}

inline  Standard_Real IntCurvesFace_NewShapeIntersector::WParameter(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->WParameter(IndexIntPnt(IndexPt(i))));
}

inline  const gp_Pnt& IntCurvesFace_NewShapeIntersector::Pnt(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->Pnt(IndexIntPnt(IndexPt(i))));
}

inline  IntCurveSurface_TransitionOnCurve  IntCurvesFace_NewShapeIntersector::Transition(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->Transition(IndexIntPnt(IndexPt(i))));
}

inline  TopAbs_State  IntCurvesFace_NewShapeIntersector::State(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->State(IndexIntPnt(IndexPt(i))));
}


inline  const TopoDS_Face&  IntCurvesFace_NewShapeIntersector::Face(const Standard_Integer i) const { 
  IntCurvesFace_NewIntersector *Ptr = (IntCurvesFace_NewIntersector *)(PtrIntersector(IndexFace(IndexPt(i))));
  return(Ptr->Face());
}

inline Standard_Boolean IntCurvesFace_NewShapeIntersector::IsDone() const {
  return(done);
}




#endif
