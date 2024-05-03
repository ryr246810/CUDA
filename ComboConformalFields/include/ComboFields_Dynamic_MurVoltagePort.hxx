#ifndef MURVOLTAGEPORT
#define MURVOLTAGEPORT

#include <ComboFields_Dynamic_SrcBase.hxx>
#include <TxMaker.h>
#include <TxMakerMap.h>
#include <TxMakerMapBase.h>
#include <TFunc.hxx>
#include <TxHierAttribSet.h>
#include <PhysConsts.hxx>



class ComboFields_Dynamic_MurVoltagePort : public ComboFields_Dynamic_SrcBase{
public:
	PortData m_MurPort;
	double m_VBar;
	double m_Step;
	//fstream fout;

	vector<GridEdgeData*> m_FreeSpaceEdgeDatas;
	vector<GridEdgeData*> m_MurPortEdgeDatas;

	vector<GridVertexData*> m_MurPortSweptEdgeDatas;
  	vector<GridVertexData*> m_FreeSpaceSweptEdgeDatas;

	vector<double> m_amp;
	// vector<GridVertexData*> m_MurPortSweptEdgeDatas;
	// vector<GridVertexData*> m_FreeSpaceSweptEdgeDatas;
	
	TFunc* m_tfuncPtr;
	
	
	~ComboFields_Dynamic_MurVoltagePort(){
		delete m_tfuncPtr;
		//fout.close();
	}
	
	
	void SetAttrib(const TxHierAttribSet& tha);
	
	void Advance()
	{
		DynObj::Advance();
	}
	
	void Setup();
	
	
	void Advance_SI_Elec_1(const double si_scale);
        void Advance_SI_Elec_Damping_1(const double si_scale, double damping_scale);
		void Get_Parameters(Standard_Real& Ebar, Standard_Real& Ebar2);
		void Get_VBar(Standard_Real& VBar);
		void Get_amp(Standard_Real** amp, Standard_Integer& amp_size);
		void Get_Ptr(vector<GridEdgeData*>* MurEdgeDatas, vector<GridEdgeData*>* FreeEdgeDatas,
					 vector<GridVertexData*>* MurSweptEdgeDatas, vector<GridVertexData*>* FreeSweptEdgeDatas);
		void advance(Standard_Real scale);
	
	void ZeroPhysDatas(){
		int ne = m_MurPortEdgeDatas.size();
		for(int i=0; i < ne; i++){
			m_MurPortEdgeDatas[i]->ZeroPhysDatas();
		}
	}

	void SetupVP(){
		double theVp =  1.0;
		m_VBar = theVp*mksConsts.c/m_Step;
	}


	void SetupGridEdgeDatasEfficientLength(){
		int ne = m_MurPortEdgeDatas.size();
		for(int i=0; i<ne; i++){
			m_MurPortEdgeDatas[i]->ComputeEfficientLength();
		}
	}
	
	void SetupDataEdgeDatas();
  	void SetupDataSweptEdgeDatas();
	
	void setup_amp();
	
};



#endif
