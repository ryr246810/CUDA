#include <ComboFields_Dynamic_MurVoltagePort.hxx>
#include <SI_SC_Mur_Equation.hxx>


 fstream fout;
 double E0;

void ComboFields_Dynamic_MurVoltagePort::SetAttrib(const TxHierAttribSet& tha){
	//cout<<"this is ComboFields_Dynamic_MurVoltagePort "<<endl;
	
	const GridBndData* theGridBndDatas = GetFldsDefCntr()->GetGridGeom(m_PhiIndex)->GetGridBndDatas();
	int thePortIndex;
	if(tha.hasOption("mask")){
		int tmpMask = tha.getOption("mask");
		theGridBndDatas->ConvertFaceMasktoIndex(tmpMask, thePortIndex);
		
		//cout<<"mask = "<<tmpMask<<"  ,  index = "<<thePortIndex<<endl;
		
	}else{
		cout<<"EigenPortToolBase::::SetAttrib-----------error----------1"<<endl;
		exit(1);
	}
	m_MurPort = *(theGridBndDatas->GetPortWithPortIndex(thePortIndex) );
	
	// cout<<"MurPort info : "<<endl;
	// cout<<"port type "<<m_MurPort.m_Type<<endl;
	// cout<<"Dir "<<m_MurPort.m_Dir<<endl;
	// cout<<"relative dir "<<m_MurPort.m_RelativeDir<<endl;
	// cout<<"LD cord "<<m_MurPort.m_LDCords[0]<<" "<<m_MurPort.m_LDCords[1]<<endl;
	// cout<<"RU cord "<<m_MurPort.m_RUCords[0]<<" "<<m_MurPort.m_RUCords[1]<<endl;
	
	std::vector< std::string > tfuncNames = tha.getNamesOfType("TFunc");
	//cout<<"time function name is "<<tfuncNames[0]<<endl;
	if(!tfuncNames.size()){
		m_tfuncPtr = new TFunc;
		std::cout << "No temporal Function specified.\n";
	}else{
		TxHierAttribSet attribs = tha.getAttrib(tfuncNames[0]);
		string functionName = attribs.getString("function");
		try {
			m_tfuncPtr = TxMakerMap<TFunc>::getNew(functionName);
		}
		catch (TxDebugExcept& txde) {
			std::cout << txde << std::endl;
			return;
		}
		m_tfuncPtr->setAttrib(attribs);
	}
}

void ComboFields_Dynamic_MurVoltagePort::Setup(){
	fout.open("ana_value.txt", ios::out);
	
	
	//cout<<"in Mur Voltage Port setup"<<endl;
	//cout<<"m_MurPortEdgeDatas size = "<<m_MurPortEdgeDatas.size()<<endl;
	SetupDataEdgeDatas();
	SetupDataSweptEdgeDatas();
	SetupVP();
	SetupGridEdgeDatasEfficientLength();
	ZeroPhysDatas();
	setup_amp();
	//cout<<"ComboFields_Dynamic_MurVoltagePort edge num "<<m_MurPortEdgeDatas.size()<<endl;
}

void ComboFields_Dynamic_MurVoltagePort::setup_amp(){
	int ne = m_MurPortEdgeDatas.size();
	//GridEdge* BaseEdge = m_MurPortEdgeDatas[0]->GetBaseGridEdge();
	//double r_min = BaseEdge->GetFirstVertex()->GetLocation()[1];
	//double r_max = BaseEdge->GetFirstVertex()->GetLocation()[1];
	
	double r_min = m_MurPortEdgeDatas[0]->GetFirstVertex()->GetLocation()[1];
	double r_max = m_MurPortEdgeDatas[0]->GetFirstVertex()->GetLocation()[1];
	for(int i = 0; i < ne; i++){
		GridEdgeData * the_edge = m_MurPortEdgeDatas[i];
		
		///GridEdge* the_edge = m_MurPortEdgeDatas[i]->GetBaseGridEdge();
		double r1 = the_edge->GetFirstVertex()->GetLocation()[1];
		double r2 = the_edge->GetLastVertex()->GetLocation()[1];
		if(r1 < r_min){
			r_min = r1;
		}
		if(r2 < r_min){
			r_min = r2;
		}
		if(r1 > r_max){
			r_max = r1;
		}
		if(r2 > r_max){
			r_max = r2;
		}
	}
	
	//cout<<"r_min = "<<r_min<<endl;
	//cout<<"r_max = "<<r_max<<endl;
	
	
	//cout<<"compute amp"<<endl;
	for(int i = 0; i < ne; i++){
		TxVector2D<double> mid_pnt;
		m_MurPortEdgeDatas[i]->ComputeMidPntLocation(mid_pnt); //tzh Modify 20210416
		/*GridEdge* BaseEdge = m_MurPortEdgeDatas[i]->GetBaseGridEdge();
		TxVector2D<Standard_Real> firstPnt = BaseEdge->GetFirstVertex()->GetLocation();
		TxVector2D<Standard_Real> lastPnt = BaseEdge->GetLastVertex()->GetLocation();
		mid_pnt = (firstPnt + lastPnt)/2.0;*/
		double rh = mid_pnt[1];
		double b = 1.0 / log(r_max / r_min);
		double amp = -b / rh;
		m_amp.push_back(amp); 
		
		//cout<<"rh = "<<rh<<"  amp = "<<amp<<endl;
	}
	
	E0 = 1.0 / log(r_max / r_min);
	
}




void ComboFields_Dynamic_MurVoltagePort::Advance_SI_Elec_1(const double si_scale){
	DynObj::AdvanceSI(si_scale);
	
	double Dt =GetDelTime();
	double t = GetCurTime();
	double t_dalay = 0.05 / 3e8;
	double time = t - t_dalay;
	if(time < 0){
		time = 0;
	}
	double funcValue = m_tfuncPtr->operator()(time);
	double ana_value = -1.0 * funcValue * E0 / 0.0125;
	fout<<t<<" "<<ana_value<<endl;

	
	int dynEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
	int preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex();
	
	for(size_t i=0; i<m_MurPortEdgeDatas.size(); i++){
		// Advance_Mur_OneElecElem_SI_SC(m_MurPortEdgeDatas[i], m_FreeSpaceEdgeDatas[i], 
				  // Dt, si_scale,
				  // m_VBar, dynEFldIndx, preTStepEFldIndx);
		Advance_Mur_OneElecElem_SI_SC_TFunc(m_MurPortEdgeDatas[i], m_FreeSpaceEdgeDatas[i], 
				    t, Dt, si_scale,
				    m_VBar,
				    dynEFldIndx, preTStepEFldIndx,
				    m_amp[i], m_tfuncPtr);
		
	}
	for(size_t i=0; i<m_MurPortSweptEdgeDatas.size(); i++){
    		Advance_Mur_OneElecElem_SI_SC(m_MurPortSweptEdgeDatas[i], m_FreeSpaceSweptEdgeDatas[i],
                                  Dt, si_scale,
                                  m_VBar, dynEFldIndx, preTStepEFldIndx);
  }

}

void ComboFields_Dynamic_MurVoltagePort::Get_amp(Standard_Real** amp, Standard_Integer& amp_size)
{
	amp_size = m_amp.size();
	Standard_Real* tmp;
	tmp = (Standard_Real*)malloc(sizeof(Standard_Real)*amp_size);
	for(int i = 0; i < amp_size; ++i){
		tmp[i] = m_amp[i];
	}
	*amp = tmp;
}

void ComboFields_Dynamic_MurVoltagePort::advance(Standard_Real scale)
{
	DynObj::AdvanceSI(scale);
}

void ComboFields_Dynamic_MurVoltagePort::Get_Ptr(vector<GridEdgeData*>* MurEdgeDatas, vector<GridEdgeData*>* FreeEdgeDatas,
					 							 vector<GridVertexData*>* MurSweptEdgeDatas, vector<GridVertexData*>* FreeSweptEdgeDatas)
{
	*MurEdgeDatas = m_MurPortEdgeDatas;
	*FreeEdgeDatas = m_FreeSpaceEdgeDatas;
	*MurSweptEdgeDatas = m_MurPortSweptEdgeDatas;
	*FreeSweptEdgeDatas = m_FreeSpaceSweptEdgeDatas;
}

void ComboFields_Dynamic_MurVoltagePort::Get_VBar(Standard_Real& VBar)
{
	VBar = m_VBar;
}

void ComboFields_Dynamic_MurVoltagePort::Get_Parameters(Standard_Real& Ebar, Standard_Real& Ebar2)
{
	double t =  GetCurTime();
 	Standard_Real Dt = GetDelTime();

	Ebar = m_tfuncPtr->operator()(t+Dt);
	Ebar2 = m_tfuncPtr->operator()(t+Dt - 1.0/m_VBar);
}

void ComboFields_Dynamic_MurVoltagePort::Advance_SI_Elec_Damping_1(const double si_scale,double damping_scale){
  double t =  GetCurTime();
  Standard_Real Dt = GetDelTime();

  Standard_Integer dynEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_DynamicElecField_PhysDataIndex();
  Standard_Integer preTStepEFldIndx = GetFldsDefCntr()->GetFieldsDefineRules()->Get_MUR_PreStep_PhysDataIndex();

  Standard_Integer AEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_AE_PhysDataIndex();
  Standard_Integer BEIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_BE_PhysDataIndex();


	for(size_t i=0; i<m_MurPortEdgeDatas.size(); i++){
		Advance_Mur_OneElecElem_SI_SC_TFunc_Damping(m_MurPortEdgeDatas[i], m_FreeSpaceEdgeDatas[i], 
				    t, Dt, si_scale,
				    m_VBar,
				    dynEFldIndx, preTStepEFldIndx,
				    m_amp[i], m_tfuncPtr, damping_scale, AEIndex, BEIndex);
	}
	for(size_t i=0; i<m_MurPortSweptEdgeDatas.size(); i++){
		Advance_Mur_OneElecElem_SI_SC_Damping(m_MurPortSweptEdgeDatas[i], m_FreeSpaceSweptEdgeDatas[i], 
				    Dt, si_scale,
				    m_VBar,
				    dynEFldIndx, preTStepEFldIndx,
				    damping_scale, AEIndex, BEIndex);
  	}

  DynObj::AdvanceSI(si_scale);
}




void
ComboFields_Dynamic_MurVoltagePort::
SetupDataSweptEdgeDatas()
{
  TxSlab2D<Standard_Integer> theMurPortRgn;
  ComputeMurTypePortRgn(m_MurPort, theMurPortRgn);
  Standard_Integer theInterfaceGlobalIndx;
  ComputePortStartIndex(m_MurPort, theInterfaceGlobalIndx);

  Standard_Real theErrEpsilon = GetZRGrid()->GetGridLengthEpsilon();
  m_Step = GetZRGrid()->GetStep(m_MurPort.m_Dir, theInterfaceGlobalIndx);

  TxSlab2D<Standard_Integer> theRgn = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn() & theMurPortRgn;
  GetGridGeom(m_PhiIndex)->GetGridVertexDatasOfMaterialTypeOfSubRgn( (Standard_Integer)MUR,theRgn, true, m_MurPortSweptEdgeDatas);

  vector<GridVertexData*>::iterator iter;
  for(iter = m_MurPortSweptEdgeDatas.begin(); iter!= m_MurPortSweptEdgeDatas.end(); iter++){
    GridVertexData* currGridVertexData = *iter;
    TxVector2D<Standard_Real> currGridVertexDataPnt = currGridVertexData->GetLocation();
    Standard_Size refGridVertexVecIndx[2];
    currGridVertexData->GetVecIndex(refGridVertexVecIndx);
    refGridVertexVecIndx[m_MurPort.m_Dir] = theInterfaceGlobalIndx;

    Standard_Size refGridVertexScalarIndx;
    GetZRGrid()->FillVertexIndx(refGridVertexVecIndx, refGridVertexScalarIndx);

    GridVertexData* refGridVertexData = GetGridGeom(m_PhiIndex)->GetGridVertices() + refGridVertexScalarIndx;

    m_FreeSpaceSweptEdgeDatas.push_back(refGridVertexData);
  }
}
void ComboFields_Dynamic_MurVoltagePort::SetupDataEdgeDatas()
{
	TxSlab2D<Standard_Integer> theMurPortRgn;
	ComputeMurTypePortRgn(m_MurPort, theMurPortRgn);
	Standard_Integer theInterfaceGlobalIndx;
	ComputePortStartIndex(m_MurPort, theInterfaceGlobalIndx);

	Standard_Real theErrEpsilon = GetZRGrid()->GetGridLengthEpsilon();
	m_Step = GetZRGrid()->GetStep(m_MurPort.m_Dir, theInterfaceGlobalIndx);
	
	TxSlab2D<Standard_Integer> theRgn = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn() & theMurPortRgn;
	GetGridGeom(m_PhiIndex)->GetGridEdgeDatasOfMaterialTypeOfSubRgn( (Standard_Integer)MUR,theRgn, false, m_MurPortEdgeDatas);

	vector<GridEdgeData*>::iterator iter;
	for(iter = m_MurPortEdgeDatas.begin(); iter!= m_MurPortEdgeDatas.end(); iter++){
		GridEdgeData* currGridEdgeData = *iter;
		TxVector2D<Standard_Real> currGridEdgeDataMidPnt;
		currGridEdgeData->ComputeMidPntLocation(currGridEdgeDataMidPnt);

		GridEdge* currGridEdge = currGridEdgeData->GetBaseGridEdge();
		Standard_Integer dir0 = currGridEdgeData->GetDir();

		Standard_Size refGridEdgeVecIndx[2];
		currGridEdge->GetVecIndex(refGridEdgeVecIndx);
		refGridEdgeVecIndx[m_MurPort.m_Dir] = theInterfaceGlobalIndx;

		Standard_Size refGridEdgeScalarIndx;
		GetZRGrid()->FillEdgeIndx(dir0,refGridEdgeVecIndx, refGridEdgeScalarIndx);
		GridEdge* refGridEdge = GetGridGeom(m_PhiIndex)->GetGridEdges()[dir0] + refGridEdgeScalarIndx;
		vector<GridEdgeData*> refGridEdgeDatas = refGridEdge->GetEdges();

		bool isPushed = false;
		if( refGridEdgeDatas.size()>=1){
			for(size_t i=0; i<refGridEdgeDatas.size(); i++){
				TxVector2D<Standard_Real> refGridEdgeDataMidPnt;
				refGridEdgeDatas[i]->ComputeMidPntLocation(refGridEdgeDataMidPnt);
				Standard_Real theLengthErr1 = fabs(refGridEdgeDataMidPnt[dir0]-currGridEdgeDataMidPnt[dir0]);
				Standard_Real theLengthErr2 = fabs(refGridEdgeDatas[i]->GetLength()-currGridEdgeData->GetLength());

				if( (theLengthErr1<theErrEpsilon) && (theLengthErr2<theErrEpsilon) ){
					m_FreeSpaceEdgeDatas.push_back(refGridEdgeDatas[i]);
					isPushed = true;
					break;
				}else{
				}
			}

			if(!isPushed){
				cout<<"error--------------------------MurBndData is not set correctedly---------------------101"<<endl;
				exit(1);
			}
		}
	}
}
