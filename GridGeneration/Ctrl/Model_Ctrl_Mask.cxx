#include <Model_Ctrl.hxx>



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
SetShapeMask(const TopoDS_Shape & theShape, const Standard_Integer theMask)
{
  if(theShape.IsNull()){
    return;
  }
  if( !(m_ShapesWithMaskTool.IsBound(theShape)) ){
    m_ShapesWithMaskTool.Bind(theShape, theMask);
  }else{
    m_ShapesWithMaskTool.ChangeFind(theShape) = theMask;
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
EraseShapeMask(const TopoDS_Shape & theShape)
{
  if(theShape.IsNull()){
    return;
  }
  if( m_ShapesWithMaskTool.IsBound(theShape) ){
    m_ShapesWithMaskTool.UnBind(theShape);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
SetSpecialFaceMask(const TopoDS_Face& theFace, const Standard_Integer theMask)
{
  if(theFace.IsNull()){
    return;
  }
  if( !(m_SpecialFacesWithMaskTool.IsBound(theFace)) ){
    m_SpecialFacesWithMaskTool.Bind(theFace, theMask);
  }else{
    m_SpecialFacesWithMaskTool.ChangeFind(theFace) = theMask;
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
EraseSpecialFaceMask(const TopoDS_Face & theFace)
{
  if(theFace.IsNull()){
    return;
  }
  if( m_SpecialFacesWithMaskTool.IsBound(theFace) ){
    m_SpecialFacesWithMaskTool.UnBind(theFace);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
Setup_Map_Of_Shape_Mask_Index()
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_ShapesWithMaskTool); Iter.More(); Iter.Next()){

    const TopoDS_Shape & theShape = Iter.Key();
    Standard_Integer theMask = Iter.Value();

    MapMaskToShapeIndex(theShape, theMask);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void 
Model_Ctrl::
Setup_Map_Of_Face_Mask_Index()
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_SpecialFacesWithMaskTool); Iter.More(); Iter.Next()){

    const TopoDS_Face & theFace = TopoDS::Face(Iter.Key());
    Standard_Integer theMask = Iter.Value();

    MapMaskToFaceIndex(theFace, theMask);
  }
}






/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::MapMaskToShapeIndex(const TopoDS_Shape& theShape,
				     const Standard_Integer theMask)
{
  if(theShape.IsNull()) return;
  Standard_Integer theShapeIndex = GetShapeIndex(theShape);
  if(theShapeIndex==0) return;

  if( m_ShapeMaskWithIndexTool.IsBound(theMask) ){
    Standard_Integer theOldMask =  m_ShapeMaskWithIndexTool.Find(theMask);
    m_ShapeMaskWithIndexTool.ChangeFind(theMask) = theShapeIndex;
  }else{
    m_ShapeMaskWithIndexTool.Bind(theMask, theShapeIndex);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::MapMaskToFaceIndex(const TopoDS_Face& theFace,
				    const Standard_Integer theMask)
{
  if(theFace.IsNull()) return;
  Standard_Integer theFaceIndex = GetFaceIndex(theFace);
  if(theFaceIndex==0) return;

  if( m_FaceMaskWithIndexTool.IsBound(theMask) ){
    Standard_Integer theOldMask = m_FaceMaskWithIndexTool.Find(theMask);
    m_FaceMaskWithIndexTool.ChangeFind(theMask) = theFaceIndex;
  }else{
    m_FaceMaskWithIndexTool.Bind(theMask, theFaceIndex);
  }
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ResetFacesMaskDefine()
{
  m_FaceMaskWithIndexTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ResetShapesMaskDefine()
{
  m_ShapeMaskWithIndexTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ClearFacesMask()
{
  m_SpecialFacesWithMaskTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::ClearShapesMask()
{
  m_ShapesWithMaskTool.Clear();
}

