#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>

#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgo_Section.hxx>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_HCurve.hxx>

#include <BRepBuilderAPI_NurbsConvert.hxx>


#define GRID_GENERATION_DBG_2


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::SetupFaceBndPntDataMaterial(FaceBndPntData& aData)
{
  aData.MaterialType = GetModelsCtrl()->GetMaterialTypeWithShapeIndex(aData.TheShapeIndex);


  const TColStd_DataMapOfIntegerListOfInteger& theEdgeFaceTool = GetModelsCtrl()->GetEdgeWithFace();
  const TColStd_ListOfInteger& theFaceIndices = theEdgeFaceTool.Find( aData.TheEdgeIndex);

  TColStd_ListIteratorOfListOfInteger iter;

  for(iter.Initialize(theFaceIndices); iter.More();  iter.Next() ){
    Standard_Integer currFaceIndex = iter.Value();
    Standard_Integer theFaceMaterialType = GetModelsCtrl()->GetMaterialTypeWithFaceIndex( currFaceIndex );
    aData.MaterialType = aData.MaterialType | theFaceMaterialType;
  }
}



/*************************************************************************************/
void Grid_Generation::BuildFaceBndPnts()
{
  cout<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<"\t Beginning of BuildFaceBndPnts Function"<<endl;



  const TColStd_DataMapOfIntegerListOfInteger& theEdgeFaceTool = GetModelsCtrl()->GetEdgeWithFace();


  vector<FaceBndPntData>* theData = ModifyFaceBndPnt();
  theData->clear();


  // 1. Build a TopoDS_Face according GridPlane;
  TxVector<Standard_Real> aPnt = GetZRDefine()->GetXYZOrg();
  TxVector<Standard_Real> aDir = GetZRDefine()->GetWorkPlaneUnitVec();

  gp_Pln aGridPlane(gp_Pnt(aPnt[0],aPnt[1],aPnt[2]), gp_Dir(aDir[0], aDir[1], aDir[2]) );
  TopoDS_Face aFace = BRepBuilderAPI_MakeFace(aGridPlane).Face();

  // 2. intersector
  IntCurvesFace_Intersector aCF(aFace, m_Tol);
  
  // 3. the all edges of model
  const TopTools_DataMapOfShapeInteger& theAllEdges = GetModelsCtrl()->GetEdgesWithIndex();
  TopTools_DataMapIteratorOfDataMapOfShapeInteger iter;
  
  for(iter.Initialize(theAllEdges); iter.More(); iter.Next() ){
    const TopoDS_Edge& theEdge = TopoDS::Edge(iter.Key());
    const Standard_Integer& theEdgeIndex = iter.Value();
    
    if(theEdgeFaceTool.IsBound(theEdgeIndex)){
      const TColStd_ListOfInteger& theFaceIndicesOfThisEdge = theEdgeFaceTool.Find(theEdgeIndex);
      if(theFaceIndicesOfThisEdge.Extent() == 2){
	Standard_Integer theShapeIndex;
	bool isFacesBelongOneShape;
	GetModelsCtrl()->CheckAndGetShapeIndexFromFaceIndices(theFaceIndicesOfThisEdge, theShapeIndex, isFacesBelongOneShape);
	if( isFacesBelongOneShape )  {
	  // Build an HCurve according Edge
	  BRepAdaptor_Curve acurve;
	  acurve.Initialize(theEdge);
	  Handle(BRepAdaptor_HCurve) Hcurve =  new BRepAdaptor_HCurve(acurve);
	  Standard_Real PInf = Hcurve->FirstParameter();
	  Standard_Real PSup = Hcurve->LastParameter();
	  
	  // Perform the Intersection
	  aCF.Perform(Hcurve, PInf, PSup);
	  // Push all intersection pnts to the vector (V_CPS)
	  Standard_Integer nb =  aCF.NbPnt();
	  for(Standard_Integer n = 1; n<=nb; n++ ){
	    if(aCF.Transition(n)==IntCurveSurface_Tangent) continue;
	    // Build a FaceBndPntData

	    TxVector2D<Standard_Real> tmpZRPnt;
	    TxVector<Standard_Real> tmpXYZPnt(aCF.Pnt(n).X(), aCF.Pnt(n).Y(), aCF.Pnt(n).Z());
	    GetZRDefine()->Convert_XYZ_to_ZR(tmpXYZPnt, tmpZRPnt);

	    if(GetZRGrid()->IsIn(tmpZRPnt)){
	      FaceBndPntData aData;
	      aData.ThePnt = aCF.Pnt(n);
	      aData.TheShapeIndex = theShapeIndex;
	      aData.TheEdgeIndex = theEdgeIndex;
	      SetupFaceBndPntDataMaterial(aData);
	    
	      // Push an intersection pnt to the vector (V_CPS)
	      theData->push_back(aData);
	    }
	  }
	}
      }
    }else{
      cout<<"Grid_Generation::BuildFaceBndPnts---------------There is an edge with no ancestor"<<endl;
    }
  }

  cout<<"\t End of BuildFaceBndPnts Function"<<endl;
  cout<<"\t -------------------------------------------"<<endl;
  cout<<endl;
}
/*************************************************************************************/




/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::FaceBndPntConvertToFaceBndVertex()
{
  const vector<FaceBndPntData>* theCPData = GetFaceBndPnt();
  vector<FaceBndVertexData>* theCVData = m_GridBndDatas->ModifyFaceBndVertexData();
  theCVData->clear();

  Standard_Size theGridFaceIndex, theCutPlnIndex;

  Standard_Size theIndex1,theIndex2;
  Standard_Size theFrac1, theFrac2;
  bool beInRgn;

  vector<FaceBndPntData>::const_iterator iter; 

  for(iter = theCPData->begin(); iter!=theCPData->end(); iter++){
    if( Is_FaceBndPnt_OnOnePort(*iter) ){
      continue;
    }

    TxVector<Standard_Real> theXYZPnt( (iter->ThePnt).X(),  (iter->ThePnt).Y(),  (iter->ThePnt).Z() );
    Standard_Real theZRPnt[2];
    GetZRDefine()->Convert_XYZ_to_ZR(theXYZPnt, theZRPnt);

    beInRgn = false;
    GetZRGrid()->ComputeLocationOfFaceBndPnt(theZRPnt, beInRgn, theIndex1, theFrac1, theIndex2, theFrac2);

    Standard_Size theVecIndex[2];
    theVecIndex[0] = theIndex1;
    theVecIndex[1] = theIndex2;
    GetZRGrid()->FillFaceIndx(theVecIndex, theGridFaceIndex);

    if(beInRgn){
      FaceBndVertexData aFaceBndVertex;
      SetFaceBndVertexFromFaceBndPntAndItsGridLocation(*iter, theGridFaceIndex, theFrac1, theFrac2, aFaceBndVertex);
      theCVData->push_back(aFaceBndVertex);
    }else{
      cout<<"Grid_Generation::FaceBndPntConvertToFaceBndVertex--------find one corner point not locating in the global grid region"<<endl;
    }
  }
}





/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::CheckFaceBndPnts()
{
  Standard_Real theRefTol = 1.0e-3 * GetZRGrid()->GetMinStep()/((Standard_Real)GetZRGrid()->GetResolutionRatio());

  vector<FaceBndPntData>::iterator iter; 
  vector<FaceBndPntData>::iterator next_iter; 
  
  iter = m_FaceBndPnts.begin();
  while( iter != m_FaceBndPnts.end() ){
    next_iter = iter+1; 
    if( next_iter!=m_FaceBndPnts.end() ) {
      if( (*next_iter).TheEdgeIndex ==  (*iter).TheEdgeIndex ){
	Standard_Real theDistance =  ((*next_iter).ThePnt).Distance((*iter).ThePnt);
	if(theDistance < theRefTol ){
	  m_FaceBndPnts.erase(iter);
	  iter = m_FaceBndPnts.erase(next_iter);
	}else{
	  iter++;
	}
      }else{
	iter++;
      }
    }else{
      iter++;
    }
  }
}



#ifdef GRID_GENERATION_DBG_2
#undef GRID_GENERATION_DBG_2
#endif
