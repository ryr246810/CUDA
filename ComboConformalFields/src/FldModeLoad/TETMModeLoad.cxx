#include <TETMModeLoad.hxx>
#include <math.h>

TETMModeLoad::TETMModeLoad()
{
	m_GridGeometry_Cyl3D = GetGridGeom_Cyl3D();
}

TETMModeLoad::~TETMModeLoad(){

}

void TETMModeLoad::SetAttrib(const TxHierAttribSet& tha)
{
	if(tha.hasString("mode"))
	{
		m_ModeName = tha.getString("mode");
	}
	else {
		cout<<"error happens while loading mode"<<endl;
		exit(1);
	}
	
	if(tha.hasParam("freq"))
	{
		m_freq = tha.getParam("freq");
	}
	else{
		cout<<"error happens while loading frequence"<<endl;
		exit(1);
	}
	
	if(tha.hasOption("modeNumber"))
	{
		m_ModeNum = tha.getOption("modeNumber");
	}
	else{
		cout<<"error happens while loading modeNumber"<<endl;
		exit(1);
	}
	
	if(tha.hasParam("amplitude"))
	{
		m_amplitude = tha.getParam("amplitude");
	}
	else{
		cout<<"error happens while loading amplitude"<<endl;
		exit(1);
	}
	
	if(tha.hasParam("pmn"))
	{
		pmn = tha.getParam("pmn");
	}
	else{
		cout<<"error happens while loading pmn"<<endl;
		exit(1);
	}
	
	std::vector<std::string> tfuncNames = tha.getNamesOfType("TFunc");
	if(!tfuncNames.size()){
		m_tfuncPtr = new TFunc;
	}
	else{
		TxHierAttribSet attribs = tha.getAttrib(tfuncNames[0]);
		string functionName = attribs.getString("function");
		try{
			m_tfuncPtr = TxMakerMap<TFunc>::getNew(functionName);
		}
		catch (TxDebugExcept& txde){
			std::cout<<txde<<std::endl;
			return;
		}
		m_tfuncPtr->setAttrib(attribs);
	}
	
	
}

void TETMModeLoad::Setup()
{
	m_GridGeometry_Cyl3D = GetGridGeom_Cyl3D();
	SetupOneColumnrDirEdge();
	SetupOneColumnzDirEdge();
	R = 0;
	for(int i=0; i<m_rDirEdgeDatas.size(); i++)
	{
		double dr = m_rDirEdgeDatas[i]->GetLastVertex()->GetLocation()[1] - m_rDirEdgeDatas[i]->GetFirstVertex()->GetLocation()[1];
		R += dr;
	}
	cout<<R<<endl;
}

void TETMModeLoad::SetupOneColumnrDirEdge()
{
	Standard_Integer k = GetPhiIndex();
	const GridGeometry* theGridGeom = m_GridGeometry_Cyl3D->GetGridGeometry(k);
	vector<GridEdgeData*> tmpGrideEdgeDatas1, tmpGrideEdgeDatas2;
	tmpGrideEdgeDatas1.clear();
	tmpGrideEdgeDatas2.clear();
	theGridGeom->GetAllGridEdgeDatasOfPhysRgn(false, tmpGrideEdgeDatas1);
	Standard_Size nbe = tmpGrideEdgeDatas1.size();
	for(int i=0; i<nbe; i++)
	{
		if(tmpGrideEdgeDatas1[i]->GetDir() == 1){
			tmpGrideEdgeDatas2.push_back(tmpGrideEdgeDatas1[i]);
		}
	}
	vector<GridEdgeData*> rDirOneColumnEdge;
	rDirOneColumnEdge.clear();
	m_rDirEdgeDatas.clear();
	for(int i=0; i<tmpGrideEdgeDatas2.size(); i++)
	{
		Standard_Size egdeIndex[2];
		tmpGrideEdgeDatas2[i]->GetBaseGridEdge()->GetVecIndex(egdeIndex);
		if(egdeIndex[0] == 2)
		{
			GridEdgeData* currGridEdgeData = tmpGrideEdgeDatas2[i];
			rDirOneColumnEdge.push_back(currGridEdgeData);
		}
	}
	m_rDirEdgeDatas = rDirOneColumnEdge;
}

void TETMModeLoad::SetupOneColumnzDirEdge()
{
	Standard_Integer k = GetPhiIndex();
	const GridGeometry* theGridGeom = m_GridGeometry_Cyl3D->GetGridGeometry(k);
	vector<GridEdgeData*> tmpGrideEdgeDatas1, tmpGrideEdgeDatas2;
	tmpGrideEdgeDatas1.clear();
	tmpGrideEdgeDatas2.clear();
	theGridGeom->GetAllGridEdgeDatasOfPhysRgn(false, tmpGrideEdgeDatas1);
	Standard_Size nbe = tmpGrideEdgeDatas1.size();
	for(int i=0; i<nbe; i++)
	{
		if(tmpGrideEdgeDatas1[i]->GetDir() == 0){
			tmpGrideEdgeDatas2.push_back(tmpGrideEdgeDatas1[i]);
		}
	}
	vector<GridEdgeData*> zDirOneColumnEdge;
	zDirOneColumnEdge.clear();
	m_zDirEdgeDatas.clear();
	for(int i=0; i<tmpGrideEdgeDatas2.size(); i++)
	{
		Standard_Size egdeIndex[2];
		tmpGrideEdgeDatas2[i]->GetBaseGridEdge()->GetVecIndex(egdeIndex);
		if(egdeIndex[0] == 1)
		{
			GridEdgeData* currGridEdgeData = tmpGrideEdgeDatas2[i];
			zDirOneColumnEdge.push_back(currGridEdgeData);
		}
	}
	m_zDirEdgeDatas = zDirOneColumnEdge;
}


void TETMModeLoad::Advance_SI_J(const double si_scale)	
{
	exit(-1);
	double t = GetCurTime();
	double t_delay = 0.05 / 3.0e8;
	double time = t-t_delay;
	if(time <0 ) time = 0.0;
	double funcValue = m_tfuncPtr->operator()(time);
	int dynJMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
	int dynJIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
	Standard_Integer phiNum = m_GridGeometry_Cyl3D->GetDimPhi();
	if(m_ModeName == "TM")
	{
		int m = m_ModeNum/10;
		int phiNum = m_GridGeometry_Cyl3D->GetDimPhi();
		Standard_Integer k = GetPhiIndex();
		double phi = 2.0*mksConsts.pi * k / phiNum;
		for(int i=0; i<m_zDirEdgeDatas.size(); i++)
		{
			double midr = (m_zDirEdgeDatas[i]->GetLastVertex()->GetLocation()[1]+m_zDirEdgeDatas[i]->GetFirstVertex()->GetLocation()[1])*0.5;
			double zLocation = (m_zDirEdgeDatas[i]->GetLastVertex()->GetLocation()[0]+m_zDirEdgeDatas[i]->GetFirstVertex()->GetLocation()[0])*0.5;
			double mu = m_zDirEdgeDatas[i]->GetMu();
			double eps = m_zDirEdgeDatas[i]->GetEpsilon();
			double sigma = m_zDirEdgeDatas[i]->GetSigma();
			double vel = 1/sqrt(mu*eps);
			double omega = 2.0*mksConsts.pi*m_freq;
			double tmp=sqrt(1+(sigma/omega/eps)*(sigma/omega/eps));
			double beta = sqrt(mu*eps/2*(tmp+1))*omega;
			double PhysData = funcValue*m_amplitude*jn(m,pmn * midr/R)*cos(omega*time-beta*zLocation)*(sin(m*phi)+cos(m*phi));
			m_zDirEdgeDatas[i]->SetPhysData(dynJIndex, PhysData);
			
		}
	}
	DynObj::AdvanceSI(si_scale);
	
	
}


void TETMModeLoad::Advance_SI_MJ(const double si_scale)	
{
	exit(-1);
	double t = GetCurTime();
	double t_delay = 0.05 / 3.0e8;
	double time = t-t_delay;
	if(time <0 ) time = 0.0;
	double funcValue = m_tfuncPtr->operator()(time);
	int dynJMIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_JM_PhysDataIndex();
	int dynJIndex = GetFldsDefCntr()->GetFieldsDefineRules()->Get_J_PhysDataIndex();
	Standard_Integer phiNum = m_GridGeometry_Cyl3D->GetDimPhi();
	if(m_ModeName == "TE")
	{
		int m = m_ModeNum/10;
		Standard_Integer k = GetPhiIndex();
		double phi = 2.0*mksConsts.pi * (k+0.5) / phiNum;
		for(int i=0; i<m_rDirEdgeDatas.size(); i++)
		{
			double midr = (m_rDirEdgeDatas[i]->GetLastVertex()->GetLocation()[1]+m_rDirEdgeDatas[i]->GetFirstVertex()->GetLocation()[1])*0.5;
			double zLocation = (m_rDirEdgeDatas[i]->GetLastVertex()->GetLocation()[0]+m_rDirEdgeDatas[i]->GetFirstVertex()->GetLocation()[0])*0.5;
			double mu = m_rDirEdgeDatas[i]->GetMu();
			double eps = m_rDirEdgeDatas[i]->GetEpsilon();
			double sigma = m_rDirEdgeDatas[i]->GetSigma();
			double vel = 1/sqrt(mu*eps);
			double omega = 2.0*mksConsts.pi*m_freq;
			double tmp=sqrt(1+(sigma/omega/eps)*(sigma/omega/eps));
			double beta = sqrt(mu*eps/2*(tmp+1))*omega;
			double SweptPhysData = funcValue*m_amplitude*jn(m,pmn * midr/R)*cos(omega*time-beta*zLocation)*(sin(m*phi)+cos(m*phi));
			m_rDirEdgeDatas[i]->SetSweptPhysData(dynJMIndex, SweptPhysData);
		}
	}
	DynObj::AdvanceSI(si_scale);
	
}















