#ifndef MURVOLTAGEPORT
#define MURVOLTAGEPORT

#include <ComboFields_Dynamic_SrcBase.hxx>

class MurVlotagePort : public ComboFields_Dynamic_SrcBase{
public:
	vector<GridEdgeData*> m_MurPortEdgeDatas;
	vector<GridEdgeData*> m_FreeSpaceEdgeDatas;
	vector<double> m_DynElecDatas;
	
	PortData m_MurPort;
	double m_VBar;
	double m_Step;
	
	TFunc* m_tfuncPtr;
	
	void SetAttrib(const TxHierAttribSet& tha){
		const GridBndData* theGridBndDatas = GetFldsDefCntr()->GetGridGeom()->GetGridBndDatas();
		const GlobalGrid* theGG = GetFldsDefCntr()->GetGlobalGrid();
		
		int thePortIndex;
		if(tha.hasOption("mask")){
			int tmpMask = tha.getOption("mask");
			theGridBndDatas->ConvertFaceMasktoIndex(tmpMask, thePortIndex);
		}else{
			cout<<"EigenPortToolBase::::SetAttrib-----------error----------1"<<endl;
			exit(1);
		}
		
		m_MurPort = theGridBndDatas->GetPortWithPortIndex(thePortIndex) ;
		
		std::vector< std::string > tfuncNames = tha.getNamesOfType("TFunc");
		if(!tfuncNames.size()){
			m_tfuncPtr = new TFunc;;
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
	
	void setup(){
		
		SetupDataEdgeDatas();
		SetupGridEdgeDatasEfficientLength();
	}
	
	
	void SetupDataEdgeDatas(){
		TxSlab2D<int> theMurPortRgn;
		ComputeMurTypePortRgn(m_MurPort, theMurPortRgn);
		int theInterfaceGlobalIndx;
		ComputePortStartIndex(m_MurPort, theInterfaceGlobalIndx);
		double theErrEpsilon = GetZRGrid()->GetGridLengthEpsilon();
		m_Step = GetZRGrid()->GetStep(m_MurPort.m_Dir, theInterfaceGlobalIndx);
		TxSlab2D<int> theRgn = GetFldsDefCntr()->GetZRGrid()->GetPhysRgn() & theMurPortRgn;
		GetGridGeom()->GetGridEdgeDatasOfMaterialTypeOfSubRgn( (int)MUR, theRgn, false, m_MurPortEdgeDatas);
		vector<GridEdgeData*>::iterator iter;
		for(iter = m_MurPortEdgeDatas.begin(); iter!= m_MurPortEdgeDatas.end(); iter++){
			GridEdgeData* currGridEdgeData = *iter;
			TxVector2D<Standard_Real> currGridEdgeDataMidPnt;
			currGridEdgeData->ComputeMidPntLocation(currGridEdgeDataMidPnt);
			
			GridEdge* currGridEdge = currGridEdgeData->GetBaseGridEdge();
			Standard_Integer dir0 = currGridEdgeData->GetDir();

			int refGridEdgeVecIndx[2];
			currGridEdge->GetVecIndex(refGridEdgeVecIndx);
			refGridEdgeVecIndx[m_MurPort.m_Dir] = theInterfaceGlobalIndx;

			int refGridEdgeScalarIndx;
			GetZRGrid()->FillEdgeIndx(dir0,refGridEdgeVecIndx, refGridEdgeScalarIndx);
			GridEdge* refGridEdge = GetGridGeom()->GetGridEdges()[dir0] + refGridEdgeScalarIndx;
			vector<GridEdgeData*> refGridEdgeDatas = refGridEdge->GetEdges();
			
			bool isPushed = false;
			if( refGridEdgeDatas.size()>=1){
				for(size_t i=0; i<refGridEdgeDatas.size(); i++){
					TxVector2D<double> refGridEdgeDataMidPnt;
					refGridEdgeDatas[i]->ComputeMidPntLocation(refGridEdgeDataMidPnt);
					double theLengthErr1 = fabs(refGridEdgeDataMidPnt[dir0]-currGridEdgeDataMidPnt[dir0]);
					double theLengthErr2 = fabs(refGridEdgeDatas[i]->GetLength()-currGridEdgeData->GetLength());

					if( (theLengthErr1<theErrEpsilon) && (theLengthErr2<theErrEpsilon) ){
						m_FreeSpaceEdgeDatas.push_back(refGridEdgeDatas[i]);
						isPushed = true;
						break;
					}
				}
			}
			if(!isPushed){
				cout<<"error--------------------------MurBndData is not set correctedly---------------------101"<<endl;
				exit(1);
			}
		}
	}
	
	
	void SetupGridEdgeDatasEfficientLength(){
		int ne = m_MurPortEdgeDatas.size();
		for(int i=0; i<ne; i++){
			m_MurPortEdgeDatas[i]->ComputeEfficientLength();
		}
	}
	
	
	
	
	
	
	
	
}



#endif
