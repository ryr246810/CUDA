#include <Mesh_Write.hxx>


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void Mesh_Write::WriteGridLine()
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);


  Standard_Real aPnt1[2];
  Standard_Real aPnt2[2];
  TxVector<Standard_Real> theXYZ1;
  TxVector<Standard_Real> theXYZ2;

  Standard_Size VIndx[2];


  Standard_Integer Dir1;
  Standard_Integer Dir2;

  Standard_Integer NDIM = 2;
  for(Standard_Integer dir =0; dir<NDIM; dir++){

    Standard_Integer Dir1 = dir;
    Standard_Integer Dir2 = (dir+1)%2;

    Standard_Integer maxIndex = m_Data->GetZRGrid()->GetVertexDimension(Dir2);
    for(Standard_Integer index=0; index<maxIndex; index++){
      VIndx[Dir1] = 0;  // along dir1
      VIndx[Dir2] = index;

      m_Data->GetZRGrid()->GetCoord_From_VertexVectorIndx(VIndx, aPnt1);
      VIndx[Dir1] = m_Data->GetZRGrid()->GetDimension(Dir1);
      m_Data->GetZRGrid()->GetCoord_From_VertexVectorIndx(VIndx, aPnt2);
      
      m_Data->GetZRDefine()->Convert_ZR_to_XYZ(aPnt1, theXYZ1);
      m_Data->GetZRDefine()->Convert_ZR_to_XYZ(aPnt2, theXYZ2);

      BRepBuilderAPI_MakeEdge ME(gp_Pnt(theXYZ1[0],theXYZ1[1],theXYZ1[2]), 
				 gp_Pnt(theXYZ2[0],theXYZ2[1],theXYZ2[2]) );	
      if (ME.IsDone()) {	
	builder.Add(Comp,ME.Edge());
      }
    }
  }

  ostringstream sstr;
  sstr<<"MeshLine";
  sstr<<".brep";
  string s=sstr.str();

  BRepTools::Write(Comp, s.c_str());
}

