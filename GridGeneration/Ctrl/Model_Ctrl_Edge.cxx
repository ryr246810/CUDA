#include <Model_Ctrl.hxx>



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Boolean DoesIntegerListContainInteger_1(const TColStd_ListOfInteger& theList, 
						 const Standard_Integer theElement)
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
void Model_Ctrl::Setup_EdgeFaceRelation()
{
  Setup_FaceWithEdgeTool();
  Setup_EdgeWithFaceTool();
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Reset_EdgesIndex()
{
  m_FaceWithEdgeTool.Clear();
  m_EdgeWithFaceTool.Clear();

  m_EdgesWithIndexTool.Clear();
  m_IndexWithEdgesTool.Clear();
}




Standard_Boolean IsBound_MapOfShapeInteger_SameShape(const TopTools_DataMapOfShapeInteger& theMap, 
						     const TopoDS_Shape& theShape)
{
  Standard_Boolean result = Standard_False;

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(theMap); Iter.More(); Iter.Next()){
    const TopoDS_Shape & theKey = Iter.Key();
    if(  theShape.IsSame(theKey) ){
      result = Standard_True;
      break;
    }
  }
  return result;
}



Standard_Boolean Find_MapOfShapeInteger_SameShape_Index(const TopTools_DataMapOfShapeInteger& theMap, 
							const TopoDS_Shape& theShape, 
							Standard_Integer& theIndex)
{
  Standard_Boolean result = Standard_False;

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(theMap); Iter.More(); Iter.Next()){
    const TopoDS_Shape & theKey = Iter.Key();
    const Standard_Integer theValue = Iter.Value();

    if(theShape.IsSame(theKey)){
      result = Standard_True;
      theIndex = theValue;
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
void Model_Ctrl::Setup_EdgeIndex()
{
  m_EdgesWithIndexTool.Clear();
  m_IndexWithEdgesTool.Clear();

  TopExp_Explorer  Ex;
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  Standard_Integer CurrentIndx=1;
  for(Iter.Initialize(m_ShapesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Shape & theShape = Iter.Key();
    const Standard_Integer theShapeIndex = Iter.Value();

    for( Ex.Init(theShape,TopAbs_EDGE);Ex.More(); Ex.Next() ){

      if( ! IsBound_MapOfShapeInteger_SameShape(m_EdgesWithIndexTool, Ex.Current()) ){
	m_EdgesWithIndexTool.Bind(Ex.Current(), CurrentIndx);
	m_IndexWithEdgesTool.Bind(CurrentIndx, Ex.Current());
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
void Model_Ctrl::Setup_FaceWithEdgeTool()
{
  m_FaceWithEdgeTool.Clear();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_FacesWithIndexTool); Iter.More(); Iter.Next() ){
    const TopoDS_Face & theFace = TopoDS::Face(Iter.Key());
    const Standard_Integer theFaceIndex = Iter.Value();

    TColStd_ListOfInteger tmpEdgesIndices;
    tmpEdgesIndices.Clear();

    // to find all vertices on theFace
    TopExp_Explorer  Ex;
    for( Ex.Init(theFace,TopAbs_EDGE);Ex.More(); Ex.Next() ){
      Standard_Integer tmpEdgeIndex;
      if( Find_MapOfShapeInteger_SameShape_Index(m_EdgesWithIndexTool, Ex.Current(), tmpEdgeIndex) ){
	if( !DoesIntegerListContainInteger_1(tmpEdgesIndices, tmpEdgeIndex)){
	  tmpEdgesIndices.Append(tmpEdgeIndex);
	}
      }else{
	cout<<"There is an error in Model_Ctrl::Setup_Face_Edge_Relation"<<endl;
      }
    }
    if( ! (m_FaceWithEdgeTool.IsBound(theFaceIndex)) ){
      m_FaceWithEdgeTool.Bind(theFaceIndex, tmpEdgesIndices);
    }
  }

}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Model_Ctrl::Setup_EdgeWithFaceTool()
{
  m_EdgeWithFaceTool.Clear();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_EdgesWithIndexTool); Iter.More(); Iter.Next() ){
    TColStd_ListOfInteger tmpFacesIndices;
    tmpFacesIndices.Clear();

    const Standard_Integer theEdgeIndex = Iter.Value();

    TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter;

    for( tmpIter.Initialize(m_FaceWithEdgeTool); tmpIter.More(); tmpIter.Next() ){
      const Standard_Integer tmpFaceIndex = tmpIter.Key();
      const TColStd_ListOfInteger& tmpEdgesIndexList = tmpIter.Value();

      if(  DoesIntegerListContainInteger_1(tmpEdgesIndexList, theEdgeIndex)  ){
	if(  !DoesIntegerListContainInteger_1(tmpFacesIndices,tmpFaceIndex)  ){
	  tmpFacesIndices.Append(tmpFaceIndex);
	}
      }
    }


    if( ! (m_EdgeWithFaceTool.IsBound(theEdgeIndex))  ){
      m_EdgeWithFaceTool.Bind(theEdgeIndex, tmpFacesIndices);
    }

  }
}



void Model_Ctrl::Write_Edges()
{
  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  for(Iter.Initialize(m_EdgesWithIndexTool); Iter.More(); Iter.Next() ){

    const TopoDS_Shape & theEdge = Iter.Key();
    const Standard_Integer theEdgeIndex = Iter.Value();

    ostringstream sstr;
    sstr<<"Sub_Edge_";
    sstr<<theEdgeIndex;
    sstr<<".brep";
    string s=sstr.str();
    
    BRepTools::Write(theEdge,s.c_str()); 
  }

  cout<<"------------------------------------------------------------------------"<<endl;

  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter1;

  for(tmpIter1.Initialize(m_FaceWithEdgeTool); tmpIter1.More(); tmpIter1.Next() ){
    const Standard_Integer tmpFaceIndex = tmpIter1.Key();
    cout<<"FaceIndex\t=\t"<<tmpFaceIndex<<endl;
    const TColStd_ListOfInteger& tmpEdgeIndexList = tmpIter1.Value();

    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(tmpEdgeIndexList); listIter.More(); listIter.Next() ){
      cout<<"\t"<<"EdgeIndex\t=\t"<<listIter.Value()<<endl;
    }
  }



  TColStd_DataMapIteratorOfDataMapOfIntegerListOfInteger tmpIter;

  for(tmpIter.Initialize(m_EdgeWithFaceTool); tmpIter.More(); tmpIter.Next() ){
    const Standard_Integer tmpEdgeIndex = tmpIter.Key();
    cout<<"EdgeIndex\t=\t"<<tmpEdgeIndex<<endl;
    const TColStd_ListOfInteger& tmpFaceIndexList = tmpIter.Value();

    TColStd_ListIteratorOfListOfInteger listIter;
    for( listIter.Initialize(tmpFaceIndexList); listIter.More(); listIter.Next() ){
      cout<<"\t"<<"FaceIndex\t=\t"<<listIter.Value()<<endl;
    }
  }


}
