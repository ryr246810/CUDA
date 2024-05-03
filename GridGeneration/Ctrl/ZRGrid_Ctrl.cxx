#include <ZRGrid_Ctrl.hxx>
#include <BaseDataDefine.hxx>
#include <PortDataFunc.hxx>
//#define MESH_CTRL_DBG


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
ZRGrid_Ctrl::
ZRGrid_Ctrl(UnitsSystemDef* theUnitsSystem, 
	  Model_Ctrl* theModelCtrl)
{
  m_UnitsSystem = theUnitsSystem;
  m_ModelsCtrl=theModelCtrl;

  m_ZRGrid=NULL;
  m_ZRDefine=NULL;

  m_GeomResolutionRatio = 0;

  m_MinWaveLength = 0.0;
  m_WaveLengthIsSet = false;

  m_ExtendedNum = 0;
  m_Margin = 0;

  for(Standard_Integer i=0; i<3; i++){
    m_LowerBndsIsSetAsPort[i] = false;
    m_UpperBndsIsSetAsPort[i] = false;
    
    m_LowerBndsIsSetAsInputPort[i] = false;
    m_UpperBndsIsSetAsInputPort[i] = false;
  }
}


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
ZRGrid_Ctrl::
~ZRGrid_Ctrl()
{
  ClearGridTools();
  if(m_ZRGrid!=NULL) delete m_ZRGrid;
  if(m_ZRDefine!=NULL) delete m_ZRDefine;
}


void 
ZRGrid_Ctrl::
ClearGridTools()
{
  map<Standard_Integer, Grid_Tool*, less<Standard_Integer> >::iterator iter;
  for(iter=m_GridTools.begin(); iter!=m_GridTools.end(); iter++){
    Grid_Tool* tmpPtr = iter->second;
    iter->second = NULL;
    delete tmpPtr;
  }
  m_GridTools.clear();
}


void 
ZRGrid_Ctrl::
SetMinWaveLength(const Standard_Real _minwavelenght)
{
  m_MinWaveLength = _minwavelenght;
  m_WaveLengthIsSet = true;
}


void 
ZRGrid_Ctrl::
SetGeomResolutionRatio( const Standard_Integer theResolution)  
{
  m_GeomResolutionRatio = theResolution;
}; // for grid resolution ratio


void 
ZRGrid_Ctrl::
SetExtendedNum(Standard_Integer theExtendedNum) 
{
  m_ExtendedNum = theExtendedNum;
};



const Model_Ctrl* 
ZRGrid_Ctrl::
GetModelsCtrl() const
{
  return m_ModelsCtrl;
};


ZRGrid* 
ZRGrid_Ctrl::
GetZRGrid() const
{
  return m_ZRGrid;
};




ZRDefine* 
ZRGrid_Ctrl::
GetZRDefine() const
{
  return m_ZRDefine;
};



const TxSlab<Standard_Real>& 
ZRGrid_Ctrl::
GetBndBox() const 
{
  return m_BndBox;
};


const Standard_Integer 
ZRGrid_Ctrl::
GetExtendedNum()const 
{
  return m_ExtendedNum;
};


const Standard_Integer 
ZRGrid_Ctrl::
GetGeomResolutionRatio() const
{
  return m_GeomResolutionRatio;
};


const Standard_Real 
ZRGrid_Ctrl::
GetMinWaveLength() const
{
  return m_MinWaveLength;
};


const Standard_Integer 
ZRGrid_Ctrl::
GetMargin() const 
{
  return m_Margin;
};


void  
ZRGrid_Ctrl::
SetMargin(const Standard_Integer _margin) 
{
  m_Margin = _margin;
};


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void 
ZRGrid_Ctrl::
CheckMeshCtrl(bool& IsDefinedProperly)
{

}


Standard_Integer 
ZRGrid_Ctrl::
GetZDir() const
{
  return m_ZDir;
}

Standard_Integer 
ZRGrid_Ctrl::
GetRDir() const
{
  return m_RDir;
}

Standard_Integer 
ZRGrid_Ctrl::
GetWorkPlaneDir() const
{
  return m_WorkPlaneDir;
}



#ifdef MESH_CTRL_DBG
#undef MESH_CTRL_DBG
#endif
