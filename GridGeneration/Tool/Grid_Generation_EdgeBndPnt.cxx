#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>

#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgo_Section.hxx>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_HCurve.hxx>


//#define MESH1_DEBUG




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::SetupEdgeBndPntDataMaterial(EdgeBndPntData& aData)
{
  Standard_Integer theShapeMaterialType =  GetModelsCtrl()->GetMaterialTypeWithShapeIndex(aData.TheShapeIndex);
  Standard_Integer theFaceMaterialType  =  GetModelsCtrl()->GetMaterialTypeWithFaceIndex(aData.TheFaceIndex);
  aData.MaterialType = theShapeMaterialType | theFaceMaterialType;
}


/****************************************************************/
/*
 * Function : BuildEdgeBndPnts
 * Purpose  : This function call BuildEdgeBndPnts(const GridLineDir) function, 
 *              and generate all grid mesh of all direction.
 */
/****************************************************************/
void Grid_Generation::BuildEdgeBndPnts()
{
  cout<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<"\t Beginning of BuildEdgeBndPnts Function"<<endl;
  for(Standard_Integer i=0;i<2;i++){
    cout<<"\t\t Building Mesh of Dir\t"<<i<<endl;
    BuildEdgeBndPnts( ZRGridLineDir(i) );
  }
  cout<<"\t End of BuildEdgeBndPnts Function"<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<endl;
}


void Grid_Generation::BuildEdgeBndPnts(const ZRGridLineDir aDir)
{
/******************************************/
 
 //stringstream ss;
 // ss<<"bnd_pnt_";
 // ss<<aDir<<".txt";
 // fstream fout;
 // char buff[300];
 // fout.open(ss.str(), ios::out);
/******************************************/
  
  Standard_Integer Dir0 = (Standard_Integer(aDir)+0)%2;
  Standard_Integer Dir1 = (Standard_Integer(aDir)+1)%2;

  Standard_Size NMAX1 = GetZRGrid()->GetVertexDimension( Dir1 );
  Standard_Size SIZE1 = GetZRGrid()->GetVertexSize( Standard_Integer(Dir1) );

  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> >* theData = ModifyEdgeBndPntOf(aDir);

  TxVector<Standard_Real> aXYZUnitVec = GetZRDefine()->GetXYZUnitVecAccordingRZDir((Standard_Integer)aDir);
  gp_Dir theDir(aXYZUnitVec[0], aXYZUnitVec[1], aXYZUnitVec[2]);

  gp_Lin aGridLine;
  TxVector<Standard_Real> aPnt;
  aPnt = GetZRDefine()->GetXYZOrg();

  aGridLine.SetDirection(theDir);
  aGridLine.SetLocation( gp_Pnt(aPnt[0],aPnt[1],aPnt[2]) );

  const TopTools_DataMapOfShapeInteger& theAllShapes = GetModelsCtrl()->GetShapesWithIndex();

  TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;

  Standard_Size indxVec[2];
  indxVec[Dir0] = 0;

  for(Standard_Size i1=0; i1<NMAX1; i1++){
    indxVec[Dir1] = i1;

    Standard_Size TheIndex = Standard_Size(i1)*SIZE1;

    vector<EdgeBndPntData> v_PS;
    v_PS.clear();

    for(Iter.Initialize(theAllShapes); Iter.More(); Iter.Next() ){
      const TopoDS_Shape& theShape = Iter.Key();
      const Standard_Integer& theShapeIndex = Iter.Value(); 

      //*
      BRepMesh_IncrementalMesh Inc(theShape, 0.1, Standard_True, 0.1 ,Standard_False, Standard_False);
      Inc.Perform();
      TopoDS_Shape theNewShape = Inc.Shape();
      m_ShapeIntersector.Load(theNewShape, m_Tol);
      //*/

      //m_ShapeIntersector.Load(theShape, m_Tol);

      Standard_Real theZRPnt[2];
      Standard_Real theStart = -1.0*GetZRGrid()->GetMinStep();
      Standard_Real theEnd = GetZRGrid()->GetLength(aDir) + 2.0*GetZRGrid()->GetMinStep();

      GetZRGrid()->GetCoord_From_VertexVectorIndx(indxVec, theZRPnt);
      GetZRDefine()->Convert_ZR_to_XYZ(theZRPnt, aPnt);

      aGridLine.SetLocation( gp_Pnt(aPnt[0],aPnt[1],aPnt[2]) );
      m_ShapeIntersector.Perform(aGridLine, theStart, theEnd);

      m_ShapeIntersector.SortResult ();
      
      Standard_Size nb = m_ShapeIntersector.NbPnt();
      for(Standard_Size n =1;n<=nb;n++)  {
	
	const TopoDS_Face& aFace = m_ShapeIntersector.Face(n);
	
	const gp_Pnt2d aPntOnSurface(m_ShapeIntersector.UParameter(n), m_ShapeIntersector.VParameter(n));
	m_Classifier.Perform(aFace, aPntOnSurface, m_Tol);
	TopAbs_State aState = m_Classifier.State();
	
	if(aState!=TopAbs_IN && aState!=TopAbs_ON){
	  TopAbs::Print(aState,cout);
	  cout<<endl;
	  continue;
	}
	
	TxVector2D<Standard_Real> tmpZRPnt;
	TxVector<Standard_Real> tmpXYZPnt(m_ShapeIntersector.Pnt(n).X(), m_ShapeIntersector.Pnt(n).Y(), m_ShapeIntersector.Pnt(n).Z());
	GetZRDefine()->Convert_XYZ_to_ZR(tmpXYZPnt, tmpZRPnt);

	if(GetZRGrid()->IsIn(tmpZRPnt)){
	  EdgeBndPntData aData;
	  aData.TheShapeIndex = theShapeIndex;
	  aData.TheFaceIndex = GetModelsCtrl()->GetFaceIndex(aFace);
	  aData.ThePnt = m_ShapeIntersector.Pnt(n);
	  aData.TransitionType = m_ShapeIntersector.Transition(n);
	  aData.StateType = m_ShapeIntersector.State(n);
	  SetupEdgeBndPntDataMaterial(aData);
	  v_PS.push_back(aData);
	  
	  /******************************************/
	  //sprintf(buff, "%30.8f %30.8f %30.8f", aData.ThePnt.X(), aData.ThePnt.Y(), aData.ThePnt.Z());
	  //fout<<buff<<endl;
	  
	  /******************************************/
	  
	  
	}
      }
    }
    if(v_PS.size()>0){
      theData->insert( pair<Standard_Size, vector<EdgeBndPntData> >(TheIndex, v_PS) );
    }
  }//i1
  
  //fout.close();
  
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::EdgeBndPntConvertToEdgeBndVertex()
{
  cout<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<"\t Begin to convert BndPnt to BndVertex"<<endl;
  cout<<"\t\t Building Mesh of Dir\t DIRZ"<<endl;
  EdgeBndPntConvertToEdgeBndVertex(DIRRZZ);
  cout<<"\t\t Building Mesh of Dir\t DIRR"<<endl;
  EdgeBndPntConvertToEdgeBndVertex(DIRRZR);
  cout<<"\t\t End of EdgeBndPntConvertToEdgeBndVertex"<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<endl;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
//*
void Grid_Generation::EdgeBndPntConvertToEdgeBndVertex(const ZRGridLineDir aDir)
{
  const map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> >* thePData = GetEdgeBndPntOf(aDir);

  map<Standard_Size,vector<EdgeBndVertexData>,less<Standard_Size> >* theVData = m_GridBndDatas->ModifyEdgeBndVertexDataOf(aDir);
  theVData->clear();

  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> >::const_iterator it1;
  vector<EdgeBndPntData>::const_iterator it2;

  Standard_Size    theIndex = 0;
  bool             beInRgn = false;
  Standard_Size    theGridEdgeIndex = 0;
  Standard_Size    theFrac = 0;

  for(it1 = thePData->begin(); it1!=thePData->end(); it1++){
    vector<EdgeBndVertexData> aVectorV;
    aVectorV.clear();
    for(it2 = (it1->second).begin(); it2 != (it1->second).end(); it2++){
      if( Is_EdgeBndPnt_OnOnePort(*it2) ){
	continue;
      }
      TxVector<Standard_Real> thePnt(it2->ThePnt.X(), it2->ThePnt.Y(), it2->ThePnt.Z());
      Standard_Real theZRPnt[2];
      GetZRDefine()->Convert_XYZ_to_ZR(thePnt, theZRPnt);
      GetZRGrid()->ComputeLocationOfEdgeBndPnt(aDir, theZRPnt, beInRgn, theGridEdgeIndex, theFrac);

      if( beInRgn ) {
	EdgeBndVertexData aBndVertex;
	if(it2->TransitionType == IntCurveSurface_Tangent) continue;
	SetEdgeBndVertexFromEdgeBndPntAndItsGridLocation(*it2, theGridEdgeIndex, theFrac, aBndVertex);
	aVectorV.push_back(aBndVertex);
      }else{
	cout<<"Grid_Generation::EdgeBndPntConvertToEdgeBndVertex--------find one Intsection point not locating in the global grid region"<<endl;
      }
    }
    if(aVectorV.size()>0){
      theIndex = it1->first;
      theVData->insert( pair<Standard_Size, vector<EdgeBndVertexData> >(theIndex, aVectorV) );
    }
  }
}
//*/


#ifdef MESH1_DBG
#undef MESH1_DBG
#endif
