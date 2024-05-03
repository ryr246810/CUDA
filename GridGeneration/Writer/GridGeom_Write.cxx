#include <GridGeom_Write.hxx>

#include <AppendingVertexDataOfGridEdge.hxx>
#include <AppendingVertexDataOfGridFace.hxx>
#include <AppendingEdgeData.hxx>

#include <OCCInclude.hxx>

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
GridGeom_Write::GridGeom_Write()
{
  m_Data = NULL;
  m_ZRDefine = NULL;
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
GridGeom_Write::GridGeom_Write(GridGeometry* _Data, ZRDefine* _zrdefine)
{
  m_Data = _Data;
  m_ZRDefine = _zrdefine;
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
GridGeom_Write::~GridGeom_Write()
{
}




/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
GridGeom_Write::
WriteInShapeGridVertices(const Standard_Integer shapeMask)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  Standard_Integer shapeIndex = 0;
  m_Data->GetGridBndDatas()->ConvertShapeMasktoIndex(shapeMask, shapeIndex);

  if(shapeIndex<1){
    return;
  }

  GridVertexData* theGVPtr =  m_Data->GetGridVertices();
  Standard_Size theGVSize = m_Data->GetVertexSize();

  for(Standard_Size i=0; i<theGVSize; i++){
    if(theGVPtr[i].HasShapeIndex(shapeIndex)){
      TxVector2D<Standard_Real> theZRPnt = theGVPtr[i].GetLocation();
      TxVector<Standard_Real> theXYZ;
      
      m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt, theXYZ);
      
      BRepBuilderAPI_MakeVertex MV( gp_Pnt(theXYZ[0],theXYZ[1],theXYZ[2]) ); 
      if (MV.IsDone())  builder.Add(Comp,MV.Vertex());
    }
  }

  ostringstream sstr;
  sstr<<"GridVertices";
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}






/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
GridGeom_Write::
WriteInShapeGridEdgeDatas(const Standard_Integer shapeMask, const Standard_Integer dir)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  Standard_Integer shapeIndex = 0;
  m_Data->GetGridBndDatas()->ConvertShapeMasktoIndex(shapeMask, shapeIndex);

  if(shapeIndex<1){
    return;
  }

  GridEdge* theGEPtr = m_Data->GetGridEdges()[dir];
  Standard_Size theGESize = m_Data->GetEdgeSize(dir);

  for(Standard_Size i=0; i<theGESize; i++){
    const vector<GridEdgeData*>& theEdgeDatas = theGEPtr[i].GetEdges();

    Standard_Size ned = theEdgeDatas.size();

    for(Standard_Size j=0; j<ned; j++){
      if(theEdgeDatas[j]->HasShapeIndex(shapeIndex)){
	TxVector2D<Standard_Real> theZRPnt1 = theEdgeDatas[j]->GetFirstVertex()->GetLocation();
	TxVector2D<Standard_Real> theZRPnt2 = theEdgeDatas[j]->GetLastVertex()->GetLocation();

	TxVector<Standard_Real> theXYZ1;
	TxVector<Standard_Real> theXYZ2;

	m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt1, theXYZ1);
	m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt2, theXYZ2);
	BRepBuilderAPI_MakeEdge ME(gp_Pnt(theXYZ1[0],theXYZ1[1],theXYZ1[2]), 
				   gp_Pnt(theXYZ2[0],theXYZ2[1],theXYZ2[2]) );	
	if (ME.IsDone()) {	
	  builder.Add(Comp,ME.Edge());
	}
      }
    }
  }

  ostringstream sstr;
  sstr<<"GridEdgeDatas_";
  sstr<<dir;
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}







void 
GridGeom_Write::
WriteGridEdges(const Standard_Integer dir)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  GridEdge* theGEPtr = m_Data->GetGridEdges()[dir];
  Standard_Size theGESize = m_Data->GetEdgeSize(dir);

  for(Standard_Size i=0; i<theGESize; i++){

    TxVector2D<Standard_Real> theZRPnt1 = theGEPtr[i].GetFirstVertex()->GetLocation();
    TxVector2D<Standard_Real> theZRPnt2 = theGEPtr[i].GetLastVertex()->GetLocation();
    
    TxVector<Standard_Real> theXYZ1;
    TxVector<Standard_Real> theXYZ2;
    
    m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt1, theXYZ1);
    m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt2, theXYZ2);
    BRepBuilderAPI_MakeEdge ME(gp_Pnt(theXYZ1[0],theXYZ1[1],theXYZ1[2]), 
			       gp_Pnt(theXYZ2[0],theXYZ2[1],theXYZ2[2]) );	
    if (ME.IsDone()) {	
      builder.Add(Comp,ME.Edge());
    }
  }

  ostringstream sstr;
  sstr<<"GridEdgeDatas_";
  sstr<<dir;
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}









void 
GridGeom_Write::
WriteAppendingVerticesOfGridEdges(const Standard_Integer dir)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  GridEdge* theGEPtr = m_Data->GetGridEdges()[dir];
  Standard_Size theGESize = m_Data->GetEdgeSize(dir);

  for(Standard_Size i=0; i<theGESize; i++){
    const vector<AppendingVertexDataOfGridEdge*>&  theAPVs = theGEPtr[i].GetAppendingVertices();
    for(Standard_Size j=0; j<theAPVs.size(); j++){
      TxVector2D<Standard_Real> aPnt = theAPVs[j]->GetLocation();
      TxVector<Standard_Real> theXYZ;
      m_ZRDefine->Convert_ZR_to_XYZ(aPnt, theXYZ);
      
      BRepBuilderAPI_MakeVertex MV( gp_Pnt(theXYZ[0],theXYZ[1],theXYZ[2]) ); 
      if (MV.IsDone())  builder.Add(Comp,MV.Vertex());
    }
  }

  ostringstream sstr;
  sstr<<"AppendingVerticesOfGridEdges_";
  sstr<<dir;
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}





void 
GridGeom_Write::
WriteAppendingVerticesOfGridFaces()
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  GridFace* theGFPtr = m_Data->GetGridFaces();
  Standard_Size theGFSize = m_Data->GetFaceSize();

  for(Standard_Size i=0; i<theGFSize; i++){
    const vector<AppendingVertexDataOfGridFace*>&  theAPVs = theGFPtr[i].GetAppendingVertices();
    for(Standard_Size j=0; j<theAPVs.size(); j++){
      TxVector2D<Standard_Real> aPnt = theAPVs[j]->GetLocation();
      TxVector<Standard_Real> theXYZ;
      m_ZRDefine->Convert_ZR_to_XYZ(aPnt, theXYZ);
      BRepBuilderAPI_MakeVertex MV( gp_Pnt(theXYZ[0],theXYZ[1],theXYZ[2]) ); 
      if (MV.IsDone())  builder.Add(Comp,MV.Vertex());
    }
  }

  ostringstream sstr;
  sstr<<"AppendingVerticesOfGridFaces_";
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}






/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
GridGeom_Write::
WriteInShapeGridFaceDatas(const Standard_Integer shapeMask)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  Standard_Integer shapeIndex = 0;
  m_Data->GetGridBndDatas()->ConvertShapeMasktoIndex(shapeMask, shapeIndex);

  if(shapeIndex<1){
    return;
  }

  GridFace* theGFPtr = m_Data->GetGridFaces();
  Standard_Size theGFSize = m_Data->GetFaceSize();

  for(Standard_Size i=0; i<theGFSize; i++){
    BRep_Builder builder1;	
    TopoDS_Compound Comp1;	
    builder.MakeCompound(Comp1);

    const vector<GridFaceData*>& theFaceDatas = theGFPtr[i].GetFaces();
    Standard_Size nfd = theFaceDatas.size();
    for(Standard_Size j=0; j<nfd; j++){
      if(theFaceDatas[j]->HasShapeIndex(shapeIndex)){
	const vector<T_Element>& theTEdges = theFaceDatas[j]->GetOutLineTEdge();
	Standard_Size ned = theTEdges.size();
	for(Standard_Size k=0; k<ned; k++){
	  GridEdgeData* currEdge = (GridEdgeData*)(theTEdges[k].GetData());

	  Standard_Integer rDir = theTEdges[k].GetRelatedDir();
	  TxVector2D<Standard_Real> theZRPnt1 = currEdge->GetFirstVertex(rDir)->GetLocation();
	  TxVector2D<Standard_Real> theZRPnt2 = currEdge->GetLastVertex(rDir)->GetLocation();
	  
	  TxVector<Standard_Real> theXYZ1;
	  TxVector<Standard_Real> theXYZ2;
	  
	  m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt1, theXYZ1);
	  m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt2, theXYZ2);

	  BRepBuilderAPI_MakeEdge ME(gp_Pnt(theXYZ1[0],theXYZ1[1],theXYZ1[2]), 
				     gp_Pnt(theXYZ2[0],theXYZ2[1],theXYZ2[2]) );	
	  if (ME.IsDone()) {	
	    builder.Add(Comp,ME.Edge());
	    builder1.Add(Comp1,ME.Edge());
	  }
	}

	vector<VertexData*> theAllVertexDatas;
	theFaceDatas[j]->GetOrderedVertexDatas(theAllVertexDatas);
	for(Standard_Size k=0;k<theAllVertexDatas.size();k++){
	  TxVector2D<Standard_Real> tmpV = theAllVertexDatas[k]->GetLocation();
	  TxVector<Standard_Real> tmpVXYZ;
	  m_ZRDefine->Convert_ZR_to_XYZ(tmpV, tmpVXYZ);
	  BRepBuilderAPI_MakeVertex MV1(gp_Pnt(tmpVXYZ[0], tmpVXYZ[1], tmpVXYZ[2]));
	  if (MV1.IsDone()) {	
	    builder1.Add(Comp1,MV1.Vertex());
	  }
	}
	TxVector2D<Standard_Real> theBC = theFaceDatas[j]->GetBaryCenter();
	TxVector<Standard_Real> theBCXYZ;
	m_ZRDefine->Convert_ZR_to_XYZ(theBC, theBCXYZ);
	BRepBuilderAPI_MakeVertex MV(gp_Pnt(theBCXYZ[0], theBCXYZ[1], theBCXYZ[2]));
	if (MV.IsDone()) {	
	  //builder.Add(Comp,MV.Vertex());
	  builder1.Add(Comp1,MV.Vertex());
	}

	ostringstream sstr1;
	sstr1<<"GridFaceDatas_";
	sstr1<<i;
	sstr1<<".brep";
	string s1=sstr1.str();
	BRepTools::Write(Comp1, s1.c_str());
      }
    }
  }

  ostringstream sstr;
  sstr<<"GridFaceDatas";
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}




















void 
GridGeom_Write::
WriteAppendingEdgeDatasOfGridFaceDatasInShape(const Standard_Integer shapeMask)
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  Standard_Integer shapeIndex = 0;
  m_Data->GetGridBndDatas()->ConvertShapeMasktoIndex(shapeMask, shapeIndex);

  if(shapeIndex<1){
    return;
  }

  GridFace* theGFPtr = m_Data->GetGridFaces();
  Standard_Size theGFSize = m_Data->GetFaceSize();

  for(Standard_Size i=0; i<theGFSize; i++){
    const vector<GridFaceData*>& theFaceDatas = theGFPtr[i].GetFaces();

    Standard_Size nfd = theFaceDatas.size();

    for(Standard_Size j=0; j<nfd; j++){
      if(theFaceDatas[j]->HasShapeIndex(shapeIndex)){

	const vector<AppendingEdgeData*>& theAEdges = theFaceDatas[j]->GetAppendingEdgeDatas();
	Standard_Size ned = theAEdges.size();
	for(Standard_Size k=0; k<ned; k++){
	  TxVector2D<Standard_Real> theZRPnt1 = theAEdges[k]->GetFirstVertex()->GetLocation();
	  TxVector2D<Standard_Real> theZRPnt2 = theAEdges[k]->GetLastVertex()->GetLocation();
	  
	  TxVector<Standard_Real> theXYZ1;
	  TxVector<Standard_Real> theXYZ2;
	  
	  m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt1, theXYZ1);
	  m_ZRDefine->Convert_ZR_to_XYZ(theZRPnt2, theXYZ2);
	  BRepBuilderAPI_MakeEdge ME(gp_Pnt(theXYZ1[0],theXYZ1[1],theXYZ1[2]), 
				     gp_Pnt(theXYZ2[0],theXYZ2[1],theXYZ2[2]) );	
	  if (ME.IsDone()) {	
	    builder.Add(Comp,ME.Edge());
	  }
	}
      }
    }
  }

  ostringstream sstr;
  sstr<<"AppendingeEdgeDatas";
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}

