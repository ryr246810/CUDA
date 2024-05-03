
#ifndef OCAF_AISShape_HeaderFile
#define OCAF_AISShape_HeaderFile

#ifndef _AIS_Shape_HeaderFile
#include <AIS_Shape.hxx>
#endif

#include <Standard_DefineHandle.hxx>



//class OCAF_AISShape :brief Interactive object, representing a vector with arrow on its end

DEFINE_STANDARD_HANDLE(OCAF_AISShape,AIS_Shape)

class OCAF_AISShape : public AIS_Shape
{
public:
  // param theShape A linear edge to be represented as a vector
  Standard_EXPORT OCAF_AISShape (const TopoDS_Shape& theShape);

  Standard_EXPORT void SetTransparency(const Standard_Real aValue);
  Standard_EXPORT void SetShadingColor(Quantity_NameOfColor aName);


  Standard_EXPORT void SetDisplayVectors(bool isShow);
  Standard_EXPORT bool isShowVectors();



  DEFINE_STANDARD_RTTIEXT(OCAF_AISShape,AIS_Shape)

protected:
  // Redefined from AIS_Shape
  virtual void Compute (const Handle(PrsMgr_PresentationManager3d)& thePresentationManager,
                        const Handle(Prs3d_Presentation)& thePresentation,
                        const Standard_Integer theMode = 0);


  void shadingMode(const Handle(PrsMgr_PresentationManager3d)& aPresentationManager,
		   const Handle(Prs3d_Presentation)& aPrs,
		   const Standard_Integer aMode);

private:
  Quantity_Color           m_ShadingColor;
  bool                     myDisplayVectors;
};

#endif
