#include <Model_Ctrl.hxx>


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Model_Ctrl::Model_Ctrl()
{
}

/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Model_Ctrl::~Model_Ctrl()
{
  ClearFacesTypeDefine();
  ClearPortsDefine();

  Reset_VerticesIndex();
  Reset_EdgesIndex();
  ResetFacesIndex();
  ResetShapesIndex();

  ClearShapesTypeDefine();


  ClearShapesMask();
  ClearFacesMask();

  ResetShapesMaskDefine();
  ResetFacesMaskDefine();
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup()
{
  AnalysisShape();

  SetupShapeIndex();
  SetupFaceIndex();
  Setup_EdgeIndex();
  Setup_VertexIndex();

  InitTypeToFace();
  SetSpecialTypeToFace();


  SetupPorts();


  Setup_VertexEdgeRelation();
  Setup_VertexFaceRelation();

  Setup_EdgeFaceRelation();

  Setup_Map_Of_Shape_Mask_Index();

  Setup_Map_Of_Face_Mask_Index();
}
