#include <GeomDataBase.hxx>


GeomDataBase::GeomDataBase()
{
  m_Mark     = ZERO_MASK;
}

GeomDataBase::GeomDataBase(Standard_Integer _mark)
{
  m_Mark = _mark;
}

GeomDataBase::~GeomDataBase()
{
}


void 
GeomDataBase::
SetupGeomDimInf()
{

}


/***************************************************************/
Standard_Integer GeomDataBase::GetMark() const
{
  return m_Mark;
}

Standard_Integer GeomDataBase::GetState() const
{
  Standard_Integer state_mark = m_Mark & STATE_ONLY_MASK;
  return state_mark;
}

Standard_Integer GeomDataBase::GetType() const
{
  Standard_Integer tmp_mark =  m_Mark & TYPE_ONLY_MASK;
  return tmp_mark;
}
/***************************************************************/



/***************************************************************/
void GeomDataBase::AddMark(Standard_Integer _mark)
{
  m_Mark |=_mark;
}

void GeomDataBase::SetState(Standard_Integer state_mark)
{
  RemoveStateMark();
  AddMark(state_mark);
}


void GeomDataBase::SetType(Standard_Integer type_mark)
{
  RemoveTypeMark();
  AddMark(type_mark);
}
/***************************************************************/



/***************************************************************/
void GeomDataBase::RemoveStateMark()
{
  m_Mark &= STATE_ZERO_MASK;
}

void GeomDataBase::RemoveTypeMark()
{
  m_Mark &= TYPE_ZERO_MASK;
}
/***************************************************************/



/***************************************************************/
void GeomDataBase::SetMark(Standard_Integer _mark)
{
  m_Mark = _mark;
}

void GeomDataBase::ResetMark()
{
  m_Mark = ZERO_MASK;
}
/***************************************************************/


Standard_Real GeomDataBase::GetGeomDim() const
{
  return 0.0;
}


Standard_Real GeomDataBase::GetDualGeomDim() const
{
  return 0.0;
}


Standard_Real GeomDataBase::GetSweptGeomDim() const
{
  return 0.0;
}


Standard_Real GeomDataBase::GetDualSweptGeomDim() const
{
  return 0.0;
}

Standard_Real GeomDataBase::GetSweptGeomDim_Near()
{
  return 0.0;
}


Standard_Real GeomDataBase::GetDualGeomDim_Near()
{
  return 0.0;
}
