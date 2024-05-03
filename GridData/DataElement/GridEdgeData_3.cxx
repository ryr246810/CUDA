
#include <GridEdge.hxx>
#include <T_Element.hxx>

#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>
#include <AppendingVertexDataOfGridEdge.hxx>

#include <GridGeometry.hxx>

#include <PhysConsts.hxx>




// From ShapeIndices
void
GridEdgeData::
DeduceGeomState()
{
  if(m_ShapeIndices.empty()){
    this->SetState(OUTSHAPE);
  }else{
    if( (this->GetFirstVertex()->GetState()==BND) && 
	(this->GetLastVertex()->GetState()==BND) ){
      this->SetState(BND);
    }else{
      this->SetState(INSHAPE);
    }
  }
}



void
GridEdgeData::
DeduceGeomType()
{
  if( (GetFirstVertex()->IsAppendedToGridEdge()) || 
      (GetLastVertex()->IsAppendedToGridEdge()) ){
    this->SetType(PFEDGE);
  }else{
    this->SetType(REGEDGE);
  }
}
