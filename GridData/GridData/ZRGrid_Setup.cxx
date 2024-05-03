#include <ZRGrid.hxx>
#include <algorithm>

//#define GG_DEBUG

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/

ZRGrid::
ZRGrid()
{
  Standard_Integer NDIM = 2;

  for(Standard_Integer i=0;i<NDIM;i++){
    m_Org[i]=0.0;
    m_Dimension[i]=0;
    m_MinSteps[i]=0.0;
  }

  m_Margin = 0;
  m_Resolution = 10;
  m_MinStep = 0.0;
  m_PhiNumber = 1;
}


void 
ZRGrid::
SetupGrid(const Standard_Real aPnt[2],
	  const vector<Standard_Real>& ZLengths,
	  const vector<Standard_Real>& RLengths,
	  const Standard_Integer theMargin,
	  const Standard_Integer thePMLLayer,
	  const Standard_Integer theResolution)
{
  SetOrg(aPnt);
  SetMargin(theMargin);
  SetPMLLayer(thePMLLayer);
  SetResolutionRatio(theResolution);

  SetupGridLengthsOf(0, ZLengths);
  SetupGridLengthsOf(1, RLengths);

  Setup();
}


void 
ZRGrid::
SetupGrid(const TxVector2D<Standard_Real>& aPnt,
	  const vector<Standard_Real>& ZLengths,
	  const vector<Standard_Real>& RLengths,
	  const Standard_Integer theMargin,
	  const Standard_Integer thePMLLayer,
	  const Standard_Integer theResolution)
{
  SetOrg(aPnt);
  SetMargin(theMargin);
  SetPMLLayer(thePMLLayer);
  SetResolutionRatio(theResolution);

  SetupGridLengthsOf(0, ZLengths);
  SetupGridLengthsOf(1, RLengths);

  Setup();
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
ZRGrid::
~ZRGrid()
{
  m_DLVectors.clear();
  m_LVectors.clear();
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
SetOrg(const TxVector2D<Standard_Real>& _aPnt)
{
  m_Org = _aPnt;
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
SetOrg(const Standard_Real _aPnt[2])
{
  for(Standard_Integer i=0;i<2;i++){
    m_Org[i]=_aPnt[i];
  }
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
SetOrg(const Standard_Real _DirCoord_0, 
       const Standard_Real _DirCoord_1)
{
  m_Org[0] = _DirCoord_0;
  m_Org[1] = _DirCoord_1;
}



void 
ZRGrid::
SetMargin(const Standard_Integer _margin)
{
  m_Margin = _margin;
}

/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
SetResolutionRatio(const Standard_Integer theResolution)
{
  m_Resolution = theResolution;
}


void 
ZRGrid::
SetPMLLayer(const Standard_Integer _pmllayer)
{
  m_PMLLayer = _pmllayer;
}




/****************************************************************/
// Function : SetupGrid
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
SetupGridLengthsOf(const Standard_Integer aDir,
		   const vector<Standard_Real>& theLengths)
{
  m_Dimension[aDir] = theLengths.size()-1;
  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::iterator iter = m_LVectors.find(aDir);
  if(iter != m_LVectors.end()){
    m_LVectors.erase(iter);
  }
  m_LVectors.insert( pair<Standard_Integer, vector<Standard_Real> > (aDir, theLengths) );
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
Setup()
{
  ComputeDimensions();
  ComputeStep();
  ComputeSecondaryParams();
  ComputeMinStep();
  ComputeMinSteps();

  ComputeRealRgn();

  /*
  vector<Standard_Real>& length0 = m_LVectors[0];
  vector<Standard_Real>& length1 = m_LVectors[1];
  cout<<"org = ["<<m_Org[0]<<", "<<m_Org[1]<<"]"<<endl;
  for(size_t i=0; i<length0.size(); i++){
    cout<<"Z["<<i<<"] = "<<length0[i]+m_Org[0]<<endl;
  }
  for(size_t i=0; i<length1.size(); i++){
    cout<<"R["<<i<<"] = "<<length1[i]+m_Org[1]<<endl;
  }
  //*/
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeStep()
{
  Standard_Integer NDIM = 2;

  m_DLVectors.clear();

  for(Standard_Integer aDir=0; aDir<NDIM; aDir++){
    const vector<Standard_Real>& theLengthVec = m_LVectors.at(aDir);
    Standard_Size nb = theLengthVec.size();
    // 1.0 make sure current dl vector is empty
    map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::iterator iter = m_DLVectors.find(aDir);
    if(iter != m_DLVectors.end()){
      m_DLVectors.erase(iter);
    }
    // 2.0 build dl vector
    vector<Standard_Real> theDLs;
    theDLs.clear();
    for(Standard_Size index=0; index<nb-1; index++){
      Standard_Real tmp = theLengthVec[index+1] - theLengthVec[index];
      theDLs.push_back(tmp);
    }
    m_DLVectors.insert( pair<Standard_Integer, vector<Standard_Real> > (aDir, theDLs));
  }
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeMinStep()
{
  Standard_Integer NDIM = 2;

  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::iterator iter;
  vector<Standard_Real>::iterator viter;
  Standard_Real tmp;

  for(Standard_Integer aDir=0; aDir<NDIM; aDir++){
    iter= m_DLVectors.find(aDir);
    viter = min_element( (iter->second).begin(), (iter->second).end() );
    if(aDir==0){
      tmp = *viter;
      m_MinStep = tmp;
    }else{
      tmp = *viter;
      m_MinStep = min(tmp, m_MinStep);
    } 
  }

  m_Tol = 1.0e-5*m_MinStep/m_Resolution;
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeMinSteps()
{
  Standard_Integer NDIM = 2;

  map<Standard_Integer, vector<Standard_Real>, less<Standard_Integer> >::iterator iter;
  vector<Standard_Real>::iterator viter;
  Standard_Real tmp;

  for(Standard_Integer aDir=0; aDir<NDIM; aDir++){
    iter= m_DLVectors.find(aDir);
    viter = min_element( (iter->second).begin(), (iter->second).end() );

    m_MinSteps[aDir] = *viter;
  }
}



void 
ZRGrid::
ComputeDimensions()
{
  m_XtndRgn.setBounds(0, 0, m_Dimension[0], m_Dimension[1]);

  m_PhysRgn = m_XtndRgn;
  m_PhysRgn.shrink(m_Margin);
}



/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeSecondaryParams()
{
  Standard_Integer NDIM = 2;

  /****************** vertexSizes ***********************/
  for(Standard_Integer i=0; i<NDIM; i++){
    m_VertexSizes[i] = 1;
    for(Standard_Integer j=i+1; j<NDIM; j++){
      m_VertexSizes[i] *= GetVertexDimension(j);
    }
  }
  m_XtndVSize = m_VertexSizes[0]*GetVertexDimension(0);
  /******************************************************/


  /********************* edgeSizes **********************/
  for(Standard_Integer aDir=0; aDir<NDIM; aDir++){
    for(Standard_Integer i=0; i<NDIM; i++){
      m_EdgeSizes[aDir][i] = 1;
      for(Standard_Integer j=i+1; j<NDIM; j++){
	m_EdgeSizes[aDir][i] *= GetEdgeDimension(aDir,j);
      }
    }


    m_XtndESize[aDir] = GetEdgeDimension(aDir, aDir);
    for(Standard_Integer i=1; i<NDIM; ++i){
      Standard_Size currDir = (aDir+i)%NDIM;
      m_XtndESize[aDir] *=  GetEdgeDimension(aDir, currDir);
    }
  }
  /******************************************************/



  /****************** faceSizes ***********************/
  for(Standard_Integer i=0; i<NDIM; ++i){
    m_FaceSizes[i] = 1;
    for(Standard_Integer j=i+1; j<NDIM; ++j){
      m_FaceSizes[i] *= GetFaceDimension(j);
    }
  }
  m_XtndFSize = m_FaceSizes[0]*GetFaceDimension(0);
  /******************************************************/
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid::
ComputeRealRgn()
{
  Standard_Real zmin = m_Org[0];
  Standard_Real rmin = m_Org[1];

  Standard_Real zmax = m_Org[0] + GetLength(0);
  Standard_Real rmax = m_Org[1] + GetLength(1);

  m_RealRgn.setBounds(zmin, rmin, zmax,rmax);
}

