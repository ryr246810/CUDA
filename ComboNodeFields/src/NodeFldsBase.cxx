#include <NodeFldsBase.hxx>

#include <NodeFlds_ConstValueRgnSetter.hxx>

#include <DynObj.hxx>

NodeFldsBase::NodeFldsBase()
{
  m_Data = NULL;
}


NodeFldsBase::NodeFldsBase(std::string nm, GridGeometry* gridGeom, size_t numComp)
{
  m_Data = NULL;
  Setup(nm, gridGeom, numComp);
}


void NodeFldsBase::Setup(std::string nm, GridGeometry* gridGeom, size_t numComp)
{
  SetName(nm);
  m_GridGeom = gridGeom;
  SetupArray(numComp);
}


// destructor
NodeFldsBase::~NodeFldsBase()
{
  if(m_Data!=NULL) delete [] m_Data;
}


Standard_Real* 
NodeFldsBase::
GetDataPtr() const
{
  return m_Data;
}


// Set up the grid and array size
void NodeFldsBase::SetupArray(size_t numComp)
{ 
  // Set size and initialize to zero
  size_t lens[3];

  // for xtndRgn element number
  for(size_t i=0; i<2; ++i){
    lens[i] = (GetZRGrid()->GetXtndRgn()).getLength(i)+1;
  }
  lens[2] = numComp;

  setLengths(lens);

  Reset();
}


void 
NodeFldsBase::
setLengths(Standard_Size lens[3])
{
  if(m_Data!=NULL) delete [] m_Data;

  for (size_t i=0; i<3; ++i) m_Lengths[i] = lens[i];

  for(Standard_Integer i=0; i<3; i++){
    m_Size[i] = 1;
    for(Standard_Integer j=i+1; j<3; j++){
      m_Size[i] *= m_Lengths[j] ;
    }
  }

  m_DataSize = m_Lengths[0]*m_Size[0];
  m_Data = new Standard_Real[m_DataSize];

  /*
  for (size_t i=0; i<3; ++i){
    cout<<"m_Lengths["<<i<<"]"<<m_Lengths[i]<<endl;
  }
  for(Standard_Integer i=0; i<3; i++){
    cout<<"m_Size["<<i<<"]"<<m_Size[i]<<endl;
  }
  cout<<"m_DataSize = "<<m_DataSize<<endl;
  //*/
}


void 
NodeFldsBase::
Add(NodeFldsBase* theOtherFldsData)
{
  Standard_Real* theOhterDataPtr = theOtherFldsData->GetDataPtr();

  for(size_t i=0; i<m_DataSize; i++){
    m_Data[i] += theOhterDataPtr[i];
  }
}


void 
NodeFldsBase::
Copy(NodeFldsBase* theOtherFldsData)
{
  Standard_Real* theOhterDataPtr = theOtherFldsData->GetDataPtr();

  for(size_t i=0; i<m_DataSize; i++){
    m_Data[i] = theOhterDataPtr[i];
  }
}


void 
NodeFldsBase::
Average(NodeFldsBase* theOtherFldsData)
{
  Standard_Real* theOhterDataPtr = theOtherFldsData->GetDataPtr();

  for(size_t i=0; i<m_DataSize; i++){
    m_Data[i] += theOhterDataPtr[i];
    m_Data[i] *= 0.5;
  }
}


//
// The various parts of the update process
//
void NodeFldsBase::Reset()
{
  for(size_t i=0; i<GetSize(); ++i) {
    *(m_Data+i) = 0.;
  }
}


//
// The various parts of the update process
//
void NodeFldsBase::Reset(size_t component)
{
  SetConstComponent(component, 0.0);
}


void 
NodeFldsBase::
SetConstComponent(size_t component,Standard_Real constValue)
{
  size_t NDIM = 2;

  NodeFlds_ConstValueRgnSetter valueSetter = NodeFlds_ConstValueRgnSetter(this);
  valueSetter.SetInitialConstValue(constValue);

  // 1. set setting rgn
  TxSlab2D<int> allocRgn = GetZRGrid()->GetXtndRgn();
  valueSetter.SetRegion(allocRgn);

  // 2. set to the component
  valueSetter.ptrReset();
  valueSetter.m_rsltIter.bump(NDIM,component);

  // 3. update zeroSetter
  valueSetter.UpdateVertices();
}


void 
NodeFldsBase::
Multiple(Standard_Real theValue)
{
  for(size_t i=0; i<GetSize(); ++i) {
    *(m_Data+i) *= theValue;
  }
}


void NodeFldsBase::Dump(NodeFlds_OutputBase& dataWriter)
{
  //
  // If a dumpPeriod parameter was specified in the input file for this field, 
  // dump only every dumpPeriod time this function is called. Otherwise, dump 
  // every time this function is called. The logic is part of DynObj so we
  // ask this parent if we should skip this call to dump. 
  // 
  if(this->ShouldWeSkipTheDump()) return;

  dataWriter.setField(this);

  dataWriter.createFldFile();
  dataWriter.createFieldData();
  dataWriter.writeField();

  dataWriter.appendFieldAttribs();
  dataWriter.appendFieldDerivedVariablesAttrib();
  dataWriter.appendFieldglobalGridAttrib();
  dataWriter.appendFieldTimeAttrib();

  dataWriter.appendFieldRunInforAttrib();

  dataWriter.closeFieldData();
  dataWriter.closeFieldFile();
}

//void NodeFldsBase::Dump_tecplot(const string& file_prefix , int step){
void NodeFldsBase::Dump_tecplot(const char * file_name){
	//stringstream ss;
	//ss<<file_prefix.c_str()<<"_"<<step<<".plt";

	double *X, *Y, *P1, *P2, *P3;
	int nvz = getLength(0);
	int nvr = getLength(1);
	int nphy = getLength(2);

	int nv = (nvr - 3) * (nvz - 3);
	X = new double[nv];
	Y = new double[nv];
	P1 = new double[nv];
	P2 = new double[nv];
	P3 = new double[nv];

	const ZRGrid * grid = m_GridGeom->GetZRGrid();

	int index = 0; 
	for(int i = 1; i < nvz - 2; i++){
		for(int j = 1; j < nvr - 2; j++){
			X[index] = grid->GetLength(1, j);
			Y[index] = grid->GetLength(0, i);
			P1[index] = m_Data[(i * nvr + j) *  nphy + 0];
			P2[index] = m_Data[(i * nvr + j) *  nphy + 1];
			P3[index] = m_Data[(i * nvr + j) *  nphy + 2];
			index++;
		}
	}
	 
	double SolTime;
	INTEGER4 Debug, I, J, III, DIsDouble, VIsDouble, IMax, JMax, KMax, ZoneType, StrandID, ParentZn, IsBlock;
	INTEGER4 ICellMax, JCellMax, KCellMax, NFConns, FNMode, ShrConn, FileType;
	INTEGER4 fileFormat = 0; 

	Debug     = 1;
	VIsDouble = 1;// double precision
	DIsDouble = 1;// double precision

	IMax      = nvr - 3;
	JMax      = nvz - 3;
	KMax      = 1;

	ZoneType  = 0;      /* Ordered */
	SolTime   = 360.0;
	StrandID  = 0;     /* StaticZone */
	ParentZn  = 0;      /* No Parent */
	IsBlock   = 1;      /* Block */
	ICellMax  = 0;
	JCellMax  = 0;
	KCellMax  = 0;
	NFConns   = 0;
	FNMode    = 0;
	ShrConn   = 0;
	FileType  = 0;

	I = TECINI142((char*)"SIMPLE DATASET",
				  (char*)"X Y P1 P2 P3",
				  (char*)file_name,
				  (char*)".",
				  &fileFormat,
				  &FileType,
				  &Debug,
				  &VIsDouble);

	I = TECZNE142((char*)"Simple Zone",
				  &ZoneType,
				  &IMax,
				  &JMax,
				  &KMax,
				  &ICellMax,
				  &JCellMax,
				  &KCellMax,
				  &SolTime,
				  &StrandID,
				  &ParentZn,
				  &IsBlock,
				  &NFConns,
				  &FNMode,
				  0,              /* TotalNumFaceNodes */
				  0,              /* NumConnectedBoundaryFaces */
				  0,              /* TotalNumBoundaryConnections */
				  NULL,           /* PassiveVarList */
				  NULL,           /* ValueLocation = Nodal */
				  NULL,           /* SharVarFromZone */
				  &ShrConn);

	III = IMax * JMax;
	I   = TECDAT142(&III, &X[0], &DIsDouble);
	I   = TECDAT142(&III, &Y[0], &DIsDouble);
	I   = TECDAT142(&III, &P1[0], &DIsDouble);
	I   = TECDAT142(&III, &P2[0], &DIsDouble);
	I   = TECDAT142(&III, &P3[0], &DIsDouble);
	I = TECEND142();

	delete[] X;
	delete[] Y;
	delete[] P1;
	delete[] P2;
	delete[] P3;

}


void NodeFldsBase::Dump_tecplot_Field_txt(string file_name, int n)
{
	int TotalFaceNum = m_GridGeom->GetFaceSize();
        int TotalZEdgeNum =  m_GridGeom->GetEdgeSize(0);
        int TotalREdgeNum =  m_GridGeom->GetEdgeSize(1);
        int rDirOneLineEdgeNum = TotalREdgeNum-TotalFaceNum;
        int zDirOneLineEdgeNum = TotalFaceNum/rDirOneLineEdgeNum;

	stringstream ss;
	ss<<"HPhi_"<<n<<".txt";
	file_name+="/"+ss.str();
	fstream fout;
	fout.open(file_name.c_str(), ios::out);
	fout<<"TITLE="<<'"'<<"HPhi"<<'"'<<endl;
	fout<<"VARIABLES = "<<'"'<<"z"<<'"'<<", "<<'"'<<"r"<<'"'<<", "<<'"'<<"HPhi"<<'"'<<endl;
	vector<Standard_Real> HPhi;
	HPhi.clear();
	for(int i=0; i<zDirOneLineEdgeNum; i++)
	{
		for(int j=0; j<rDirOneLineEdgeNum; j++)
		{
			HPhi.push_back(0.0);
		}
	}
	vector<GridFaceData*>  theFaceDatas;
	theFaceDatas.clear();
	m_GridGeom->GetAllGridFaceDatasOfPhysRgn(theFaceDatas);
	for(int i=0; i<theFaceDatas.size(); i++)
	{
		Standard_Integer currFaceIndex = theFaceDatas[i]->GetBaseGridFace()->GetIndex();
		Standard_Integer dynMFldIndx = theFaceDatas[i]->GetLocalIndex();
		HPhi[currFaceIndex] = theFaceDatas[i]->GetPhysData(dynMFldIndx);
	}	

	for(int i=0;i<zDirOneLineEdgeNum; i++)
	{
		fout<<"ZONE T = "<<'"'<<"BPhi"<<'"'<<" "<<"I="<<rDirOneLineEdgeNum+1<<", "<<"J="<<2<<", ";
		fout<<"DATAPACKING=BLOCK, VARLOCATION=([3]=CELLCENTERED)"<<endl;
		for(int j=0; j<rDirOneLineEdgeNum+1; j++)
		{
			double znode = (m_GridGeom->GetGridVertices()+i*(rDirOneLineEdgeNum+1)+j)->GetLocation()[0];
			fout<<znode<<" ";
		}
		for(int j=0; j<rDirOneLineEdgeNum+1; j++)
                {
                        double znode = (m_GridGeom->GetGridVertices()+(i+1)*(rDirOneLineEdgeNum+1)+j)->GetLocation()[0];
			fout<<znode<<" ";
                }
		fout<<"\n\n";
		for(int j=0; j<rDirOneLineEdgeNum+1; j++)
		{
			double rnode = (m_GridGeom->GetGridVertices()+i*(rDirOneLineEdgeNum+1)+j)->GetLocation()[1];
			fout<<rnode<<" ";
		}
		for(int j=0; j<rDirOneLineEdgeNum+1; j++)
                {
                        double rnode = (m_GridGeom->GetGridVertices()+(i+1)*(rDirOneLineEdgeNum+1)+j)->GetLocation()[1];
                        fout<<rnode<<" ";
                }
		fout<<"\n\n";
	
		for(int j=0; j<rDirOneLineEdgeNum; j++)
		{
			fout<<HPhi[i*rDirOneLineEdgeNum+j]<<" ";
		}
		fout<<"\n\n";
	}
	fout.close();
}
