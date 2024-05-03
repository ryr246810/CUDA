#include <Grid_GenerationBase.hxx>

/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_GenerationBase::Grid_GenerationBase()
{
  m_ZRGrid = NULL;
  m_ZRDefine = NULL;
  m_BackGroundMaterialType = 0;
};


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_GenerationBase::Grid_GenerationBase(ZRGrid* _zrg, 
					 ZRDefine* _zrd, 
					 const Standard_Integer _backgroundmaterialtype)
{
  m_ZRGrid = _zrg;
  m_ZRDefine = _zrd;
  m_BackGroundMaterialType = _backgroundmaterialtype;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Grid_GenerationBase::~Grid_GenerationBase()
{
};


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
void Grid_GenerationBase::SetBackGroundMaterialType(const Standard_Integer aType)
{
  m_BackGroundMaterialType = aType;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
Standard_Integer 
Grid_GenerationBase::
GetBackGroundMaterialType() const
{
  return m_BackGroundMaterialType;
}


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const ZRGrid* 
Grid_GenerationBase::
GetZRGrid() const 
{
  return m_ZRGrid;
};


/****************************************************************/
/*
 * Function : 
 * Purpose  : 
 */
/****************************************************************/
const ZRDefine* 
Grid_GenerationBase::
GetZRDefine() const
{
  return m_ZRDefine;
};
