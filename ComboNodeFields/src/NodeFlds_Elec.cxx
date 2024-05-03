#include <NodeFlds_Elec.hxx>


NodeFlds_Elec::
NodeFlds_Elec()
  : NodeFldsBase()
{
  m_DataSetter = NULL;
}


NodeFlds_Elec::
NodeFlds_Elec(std::string nm, GridGeometry* gridGeom, size_t numComp)
  : NodeFldsBase(nm, gridGeom, numComp)
{
  m_DataSetter = NULL;
}


NodeFlds_Elec::
~NodeFlds_Elec()
{
  if(m_DataSetter != NULL)
    delete m_DataSetter;
}


void 
NodeFlds_Elec::
SetPhysDataIndexInGridGeom(const Standard_Integer _index)
{
  m_DataIndexInGridGeom = _index;
};


void 
NodeFlds_Elec::
SetupDataSetter()
{
  m_DataSetter = new NodeFlds_Elec_RgnSetter(this);
  m_DataSetter->SetEdgePhysDataIndex(m_DataIndexInGridGeom);

  // do not to be worried the index of the datas exceed the upper bound
  TxSlab2D<int> allocRgn = this->GetZRGrid()->GetPhysRgn();
  m_DataSetter->SetRegion(allocRgn);
}


void 
NodeFlds_Elec::
Update()
{
  m_DataSetter->ptrReset();
  m_DataSetter->UpdateVertices();
  DynObj::Advance();
}

