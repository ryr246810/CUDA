#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>
#include <BaseDataDefine.hxx>


//#define MESH3_DBG
/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::BuildPort()
{
  map<Standard_Integer, PortData, less<Standard_Integer> >* thePortDatas = m_GridBndDatas->ModifyPorts();
  thePortDatas->clear();

  const TColStd_DataMapOfIntegerInteger& thePorts = GetModelsCtrl()->GetPortsWithType();
  TColStd_DataMapIteratorOfDataMapOfIntegerInteger Iter;

  Standard_Integer thePortIndex;
  Standard_Integer thePortType;
  gp_Pnt theBaryCenter;
  GridLineDir theLineDir;
  Standard_Integer theRelativeDir;

  TxSlab<Standard_Real> realXYZRgn;

  TxSlab2D<Standard_Real> realRZRgn;
  TxSlab2D<Standard_Size> gridRZRgn;

  for(Iter.Initialize(thePorts); Iter.More(); Iter.Next() ){
    thePortIndex = Iter.Key();
    thePortType = Iter.Value();

    GetModelsCtrl()->ComputePortDirWithFaceIndexOfPort(thePortIndex, theBaryCenter, theLineDir, theRelativeDir);
    GetModelsCtrl()->ComputeBndBoxOfPort(thePortIndex, realXYZRgn);
	
    GetZRDefine()->Convert_XYZ_to_ZR(realXYZRgn, realRZRgn);//三维XYZ坐标转换为二维ZR坐标
    GetZRGrid()->ComputeBndBoxInGrid(realRZRgn, gridRZRgn);
	
	
	
	
	

	

#ifdef MESH3_DBG
    cout<<"Port Region is defined as----------------------------0"<<endl;
    gridRZRgn.write(cout);
#endif

    ZRGridLineDir theZRDir = (ZRGridLineDir) GetZRDefine()->GetZRDir_AccordingTo_XYZDir(Standard_Integer(theLineDir));
    GetZRGrid()->ExtendRgnToEndAlongDir(theZRDir, theRelativeDir, gridRZRgn);// 即使MurPort也会有该操作，至少延伸到Margin层内

#ifdef MESH3_DBG
    cout<<"Port Region is defined as--------"<<endl;
    gridRZRgn.write(cout);
#endif
    
    PortData tmpPort;
    SetupOnePortData(thePortIndex, thePortType, theZRDir, theRelativeDir, gridRZRgn, tmpPort);//tmpPort中的坐标是二维的
    thePortDatas->insert( pair<Standard_Integer, PortData >(thePortIndex, tmpPort) );
	
	
	// cout<<endl;
	// cout<<"grid generation-build port :"<<endl;
	// cout<<"real XYZ rgn"<<endl;
	// realXYZRgn.writeSlab();
	
	// cout<<"real ZR rgn"<<endl;
	// realRZRgn.writeSlab();
	
	// cout<<"after extend, gridRZRgn rgn"<<endl;
	// gridRZRgn.writeSlab();
	
	
	// cout<<"grid generation port info : "<<endl;
	// cout<<"port type "<<tmpPort.m_Type<<endl;
	// cout<<"Dir "<<tmpPort.m_Dir<<endl;
	// cout<<"relative dir "<<tmpPort.m_RelativeDir<<endl;
	// cout<<"LD cord "<<tmpPort.m_LDCords[0]<<" "<<tmpPort.m_LDCords[1]<<endl;
	// cout<<"RU cord "<<tmpPort.m_RUCords[0]<<" "<<tmpPort.m_RUCords[1]<<endl;
	
	
  
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Grid_Generation::Is_EdgeBndPnt_OnOnePort(const EdgeBndPntData& theIntPnt )
{
  Standard_Integer theFaceIndex = theIntPnt.TheFaceIndex;
  bool result = GetModelsCtrl()->IsPort(theFaceIndex);
  return result;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
bool Grid_Generation::Is_FaceBndPnt_OnOnePort(const FaceBndPntData& theFaceBndPnt)
{
  bool result =  false;

  const TColStd_DataMapOfIntegerListOfInteger& theEdgeFaceTool = GetModelsCtrl()->GetEdgeWithFace();
  const TColStd_ListOfInteger& theFaceIndices = theEdgeFaceTool.Find( theFaceBndPnt.TheEdgeIndex);

  TColStd_ListIteratorOfListOfInteger iter;
  for(iter.Initialize(theFaceIndices); iter.More();  iter.Next() ){
    Standard_Integer currFaceIndex = iter.Value();
    if(GetModelsCtrl()->IsPort(currFaceIndex)){
      result = true;
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
void Grid_Generation::ExtendBndsAccordingPorts()
{
  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = m_GridBndDatas->GetPorts();
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;
  for(iter=thePorts->begin(); iter!=thePorts->end(); iter++){
    ExtendBndsAccordingPort(iter->second);
  }
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::ExtendBndsAccordingPort(const PortData& tmpPort)
{
  Standard_Integer theFaceIndex = tmpPort.m_Index;
  ZRGridLineDir thePortDir = (ZRGridLineDir)tmpPort.m_Dir;
  Standard_Integer thePortRelativeDir = tmpPort.m_RelativeDir;

  TxSlab2D<Standard_Size> thePortRgn;
  thePortRgn.setBounds(tmpPort.m_LDCords[0], 
		       tmpPort.m_LDCords[1], 
		       tmpPort.m_RUCords[0], 
		       tmpPort.m_RUCords[1]);



#ifdef MESH3_DBG
    cout<<"Grid_Generation::ExtendBndsAccordingPort----------Port Region is defined as---------0"<<endl;
    thePortRgn.write(cout);
    cout<<"GetZRGrid()->GetVertexDimension(0) = "<<GetZRGrid()->GetVertexDimension(0)<<endl;
    cout<<"GetZRGrid()->GetVertexDimension(1) = "<<GetZRGrid()->GetVertexDimension(1)<<endl;
#endif


   ExtendEdgeBndVerticesInRgn(theFaceIndex, thePortDir, thePortRelativeDir, thePortRgn);
}



void 
Grid_Generation::
CleanFaceBndVerticesAcordingPort()
{
  const map<Standard_Integer, PortData, less<Standard_Integer> >* thePorts = m_GridBndDatas->GetPorts();
  map<Standard_Integer, PortData, less<Standard_Integer> >::const_iterator iter;
  for(iter=thePorts->begin(); iter!=thePorts->end(); iter++){
    CleanFaceBndVerticesAcordingPort(iter->second);
  }
}


void 
Grid_Generation::
CleanFaceBndVerticesAcordingPort(const PortData& tmpPort)
{
  Standard_Integer theFaceIndex = tmpPort.m_Index;

  TxSlab2D<Standard_Size> thePortRgn;

  thePortRgn.setBounds(tmpPort.m_LDCords[0], 
		       tmpPort.m_LDCords[1], 
		       tmpPort.m_RUCords[0], 
		       tmpPort.m_RUCords[1]);

  CleanFaceBndVerticesInRgn(theFaceIndex, thePortRgn);
}



void 
Grid_Generation::
CleanFaceBndVerticesInRgn(const Standard_Integer thePortIndex, const TxSlab2D<Standard_Size>& theRgn)
{
  vector<FaceBndVertexData>* theFaceBndVertices = m_GridBndDatas->ModifyFaceBndVertexData();
  vector<FaceBndVertexData>::iterator viter = theFaceBndVertices->begin();

  while(viter != theFaceBndVertices->end()){
    Standard_Size theIndx = viter->m_Index;
    Standard_Size theIndxVec[2];
    GetZRGrid()->FillFaceIndxVec(theIndx, theIndxVec);

    const TColStd_ListOfInteger& theFaceIndices = (GetModelsCtrl()->GetEdgeWithFace()).Find(viter->m_EdgeIndex);
    if(theFaceIndices.Contains(thePortIndex)){
      viter = theFaceBndVertices->erase(viter);
    }else{
      viter++;
    }
  }
}




#ifdef MESH3_DBG
#undef MESH3_DBG
#endif
