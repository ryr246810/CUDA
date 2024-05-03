#include <GridGeometry_Cyl3D.hxx>

void
GridGeometry_Cyl3D::Setup()
{

  for(Standard_Size i=0;i<m_Dimphi;i++)
  {

      GridGeometry *tmp_Gridgeometry = new GridGeometry(m_ZRGrid, m_GridBndDatas);
      tmp_Gridgeometry->SetPMLDataDefine(m_PMLDefineTool);
      tmp_Gridgeometry->SetPhiIndex(i);
      tmp_Gridgeometry->SetPhiNumber(m_Dimphi);
      tmp_Gridgeometry->SetGridGeometry3D(this);
      tmp_Gridgeometry->Setup();
      m_Gridgeometry.push_back(tmp_Gridgeometry);

  }
}

void
GridGeometry_Cyl3D::Build_Near_Edge()
{

  //if(m_Dimphi < 2) return;
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
    Standard_Size index_plus=(i+1)%m_Dimphi;
    Standard_Size index_min=(i-1+m_Dimphi)%m_Dimphi;
    m_Gridgeometry[i]->SetPlusMinu_Geometry(m_Gridgeometry[index_plus],m_Gridgeometry[index_min]);

  }
  for(Standard_Size i=0;i<m_Dimphi;i++)
  {
    m_Gridgeometry[i]->Build_Near_Edge();

  }
}
