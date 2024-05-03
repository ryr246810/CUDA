#include <Model_Ctrl.hxx>



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Boolean DoesIntegerListContainInteger(const TColStd_ListOfInteger& theList, const Standard_Integer theElement)
{
  Standard_Boolean result = Standard_False;

  TColStd_ListIteratorOfListOfInteger Iter;
  for( Iter.Initialize(theList); Iter.More(); Iter.Next() ){
    if(Iter.Value() == theElement){
      result = Standard_True;
      break;
    }
  }

  return result;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_VertexFaceRelation()
{
  Setup_FaceWithVertexTool();
  Setup_VertexWithFaceTool();
}


void Model_Ctrl::Setup_VertexEdgeRelation()
{
  Setup_EdgeWithVertexTool();
  Setup_VertexWithEdgeTool();
}




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Reset_VerticesIndex()
{
  m_EdgeWithVertexTool.Clear();
  m_VertexWithEdgeTool.Clear();

  m_FaceWithVertexTool.Clear();
  m_VertexWithFaceTool.Clear();

  m_VerticesWithIndexTool.Clear();
  m_IndexWithVerticesTool.Clear();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_VertexIndex()
{
  m_VerticesWithIndexTool.Clear();
  m_IndexWithVerticesTool.Clear();


  TopExp_Explorer  Ex;
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  Standard_Integer CurrentIndx=1;
  for(Iter.Initialize(m_ShapesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();
    const Standard_Integer theShapeIndex = Iter.Value();

    for( Ex.Init(theShape,TopAbs_VERTEX);Ex.More(); Ex.Next() ){

      if( ! (m_VerticesWithIndexTool.IsBound(Ex.Current())) ){
	m_VerticesWithIndexTool.Bind(Ex.Current(), CurrentIndx);
	m_IndexWithVerticesTool.Bind(CurrentIndx, Ex.Current());
	CurrentIndx++;
      }else{
	continue;
      }

    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_FaceWithVertexTool()
{
  m_FaceWithVertexTool.Clear();


  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_FacesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Face & theFace = TopoDS::Face(Iter.Key());
    const Standard_Integer theFaceIndex = Iter.Value();

    TColStd_ListOfInteger tmpVerticesIndices;
    tmpVerticesIndices.Clear();


    // to find all vertices on theFace
    TopExp_Explorer  Ex;
    for( Ex.Init(theFace,TopAbs_VERTEX);Ex.More(); Ex.Next() ){

      if( m_VerticesWithIndexTool.IsBound(Ex.Current()) ){
	Standard_Integer tmpVertexIndex = m_VerticesWithIndexTool.Find(Ex.Current());
	
	if( !DoesIntegerListContainInteger(tmpVerticesIndices, tmpVertexIndex)){
	  tmpVerticesIndices.Append(tmpVertexIndex);
	}

      }else{
	cout<<"There is an error in Model_Ctrl::SetupVertex_Face_Relation"<<endl;
      }

    }
    if( ! (m_FaceWithVertexTool.IsBound(theFaceIndex)) ){
      m_FaceWithVertexTool.Bind(theFaceIndex, tmpVerticesIndices);
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_VertexWithFaceTool()
{
  m_VertexWithFaceTool.Clear();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_VerticesWithIndexTool); Iter.More(); Iter.Next() ){
    TColStd_ListOfInteger tmpFacesIndices;
    tmpFacesIndices.Clear();

    const Standard_Integer theVertexIndex = Iter.Value();

    TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter;

    for( tmpIter.Initialize(m_FaceWithVertexTool); tmpIter.More(); tmpIter.Next() ){
      const Standard_Integer tmpFaceIndex = tmpIter.Key();
      const TColStd_ListOfInteger& tmpVerticeIndexList = tmpIter.Value();

      if(  DoesIntegerListContainInteger(tmpVerticeIndexList, theVertexIndex)  ){
	if(  !DoesIntegerListContainInteger(tmpFacesIndices,tmpFaceIndex)  ){
	  tmpFacesIndices.Append(tmpFaceIndex);
	}
      }
    }


    if( ! (m_VertexWithFaceTool.IsBound(theVertexIndex))  ){
      m_VertexWithFaceTool.Bind(theVertexIndex, tmpFacesIndices);
    }

  }
}













/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_EdgeWithVertexTool()
{
  m_EdgeWithVertexTool.Clear();


  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_EdgesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Edge & theEdge = TopoDS::Edge(Iter.Key());
    const Standard_Integer theEdgeIndex = Iter.Value();

    TColStd_ListOfInteger tmpVerticesIndices;
    tmpVerticesIndices.Clear();


    // to find all vertices on theEdge
    TopExp_Explorer  Ex;
    for( Ex.Init(theEdge,TopAbs_VERTEX);Ex.More(); Ex.Next() ){

      if( m_VerticesWithIndexTool.IsBound(Ex.Current()) ){
	Standard_Integer tmpVertexIndex = m_VerticesWithIndexTool.Find(Ex.Current());
	
	if( !DoesIntegerListContainInteger(tmpVerticesIndices, tmpVertexIndex)){
	  tmpVerticesIndices.Append(tmpVertexIndex);
	}

      }else{
	cout<<"There is an error in Model_Ctrl::SetupVertex_Edge_Relation"<<endl;
      }

    }
    if( ! (m_EdgeWithVertexTool.IsBound(theEdgeIndex)) ){
      m_EdgeWithVertexTool.Bind(theEdgeIndex, tmpVerticesIndices);
    }
  }
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_VertexWithEdgeTool()
{
  m_VertexWithEdgeTool.Clear();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_VerticesWithIndexTool); Iter.More(); Iter.Next() ){
    TColStd_ListOfInteger tmpEdgesIndices;
    tmpEdgesIndices.Clear();

    const Standard_Integer theVertexIndex = Iter.Value();

    TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter;

    for( tmpIter.Initialize(m_EdgeWithVertexTool); tmpIter.More(); tmpIter.Next() ){
      const Standard_Integer tmpEdgeIndex = tmpIter.Key();
      const TColStd_ListOfInteger& tmpVerticeIndexList = tmpIter.Value();

      if(  DoesIntegerListContainInteger(tmpVerticeIndexList, theVertexIndex)  ){
	if(  !DoesIntegerListContainInteger(tmpEdgesIndices,tmpEdgeIndex)  ){
	  tmpEdgesIndices.Append(tmpEdgeIndex);
	}
      }
    }


    if( ! (m_VertexWithEdgeTool.IsBound(theVertexIndex))  ){
      m_VertexWithEdgeTool.Bind(theVertexIndex, tmpEdgesIndices);
    }

  }
}













void Model_Ctrl::Write_Vertices()
{
  cout<<"Model_Ctrl::Write_Vertices----------------------------------------------------------->>"<<endl;

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_VerticesWithIndexTool); Iter.More(); Iter.Next() ){

    const TopoDS_Shape & theVertex = Iter.Key();
    const Standard_Integer theVertexIndex = Iter.Value();

    ostringstream sstr;
    sstr<<"Sub_Vertex_";
    sstr<<theVertexIndex;
    sstr<<".brep";
    string s=sstr.str();
    
    BRepTools::Write(theVertex,s.c_str()); 
  
  }


  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter;

  for(tmpIter.Initialize(m_VertexWithFaceTool); tmpIter.More(); tmpIter.Next() ){
    const Standard_Integer tmpVertexIndex = tmpIter.Key();
    cout<<"VertexIndex\t=\t"<<tmpVertexIndex<<endl;
    const TColStd_ListOfInteger& tmpFaceIndexList = tmpIter.Value();

    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(tmpFaceIndexList); listIter.More(); listIter.Next() ){
      cout<<"\t"<<"FaceIndex\t=\t"<<listIter.Value()<<endl;
    }
  }


  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter1;

  for(tmpIter1.Initialize(m_FaceWithVertexTool); tmpIter1.More(); tmpIter1.Next() ){
    const Standard_Integer tmpFaceIndex = tmpIter1.Key();
    cout<<"FaceIndex\t=\t"<<tmpFaceIndex<<endl;
    const TColStd_ListOfInteger& tmpVertexIndexList = tmpIter1.Value();

    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(tmpVertexIndexList); listIter.More(); listIter.Next() ){
      cout<<"\t"<<"VertexIndex\t=\t"<<listIter.Value()<<endl;
    }
  }

  cout<<"Model_Ctrl::Write_Vertices-----------------------------------------------------------<<"<<endl;
}
