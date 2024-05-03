#include <Grid_Generation.hxx>
#include <ZRGrid_Ctrl.hxx>

#include <BRepAlgoAPI_Section.hxx>
#include <BRepAlgo_Section.hxx>


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_Generation::Grid_Generation()
  :Grid_GenerationBase()
{
  m_ModelCtrl = NULL;
  m_GridBndDatas = NULL;
  m_Tol = 1e-10;
};


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_Generation::Grid_Generation(ZRGrid* _zrg, 
				 ZRDefine* _zrd, 
				 Model_Ctrl* _model_ctrl, 
				 const Standard_Integer _backgroundmaterialtype,
				 const Standard_Real _tol)
  :Grid_GenerationBase(_zrg, _zrd, _backgroundmaterialtype)
{
  m_ModelCtrl = _model_ctrl;
  m_GridBndDatas = NULL;
  m_Tol = _tol;
};



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_Generation::~Grid_Generation()
{
  if(m_GridBndDatas!=NULL) delete m_GridBndDatas;
};



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_Generation::BuildGridBndDatas()
{
  m_GridBndDatas = new GridBndData;

  m_GridBndDatas->SetBackGroundMaterialType(GetBackGroundMaterialType());

  BuildModelsInformation();

  cout<<"grid generation build port"<<endl;
  BuildPort();

  BuildEdgeBndPnts();
  EdgeBndPntConvertToEdgeBndVertex();

  BuildFaceBndPnts();
  FaceBndPntConvertToFaceBndVertex();

  ExtendBndsAccordingPorts();
}






/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > * 
Grid_Generation::
GetEdgeBndPntOf(const ZRGridLineDir aDir) const
{
  const map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > * theData;
  switch (aDir)
    {
    case DIRRZZ:
      {
	theData = &m_EdgeBndPnts0;
	break;
      }
    case DIRRZR:
      {
	theData = &m_EdgeBndPnts1;
	break;
      }
    }
  return theData;
}



/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > * 
Grid_Generation::
ModifyEdgeBndPntOf(const ZRGridLineDir aDir)
{
  map<Standard_Size,vector<EdgeBndPntData>,less<Standard_Size> > * theData;
  switch (aDir)
    {
    case DIRRZZ:
      {
	theData = &m_EdgeBndPnts0;
	break;
      }
    case DIRRZR:
      {
	theData = &m_EdgeBndPnts1;
	break;
      }
    }
  return theData;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const vector<FaceBndPntData> * 
Grid_Generation::
GetFaceBndPnt() const
{
  return &m_FaceBndPnts;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
vector<FaceBndPntData> * 
Grid_Generation::
ModifyFaceBndPnt()
{
  return &m_FaceBndPnts;
}
