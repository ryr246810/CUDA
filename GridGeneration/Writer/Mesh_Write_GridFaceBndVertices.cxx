
#include <Mesh_Write.hxx>


/********************************************************************************************/
/*
 * Function: GetPlaneMesh
 * Property: aDir: normal dir;  aDividingFactor: dividing factor
 */
/********************************************************************************************/
void 
Mesh_Write::
WriteFaceVertices()
{
  BRep_Builder builder;	
  TopoDS_Compound Comp;	
  builder.MakeCompound(Comp);

  const vector<FaceBndVertexData>& theFaceVertices =  *(m_Data->GetGridBndDatas()->GetFaceBndVertexData());
  Standard_Size nb = theFaceVertices.size(); 
  for(Standard_Size n=0; n<nb; n++)  {
    Standard_Size tmpIndx = theFaceVertices[n].m_Index;
    Standard_Size tmpIndxVec[2];

    m_Data->GetZRGrid()->FillFaceIndxVec(tmpIndx, tmpIndxVec);

    Standard_Real tmpPnt[2];
    m_Data->GetZRGrid()->GetCoord_From_VertexVectorIndx(tmpIndxVec, tmpPnt);

    Standard_Real tmpStep0 = m_Data->GetZRGrid()->GetStep(0, tmpIndxVec[0]);
    Standard_Real tmpFrac0 = Standard_Real(theFaceVertices[n].m_Frac1)/Standard_Real( m_Data->GetZRGrid()->GetResolutionRatio() );
    tmpPnt[0] = tmpPnt[0] + tmpStep0*tmpFrac0;

    Standard_Real tmpStep1 = m_Data->GetZRGrid()->GetStep(1, tmpIndxVec[1]);
    Standard_Real tmpFrac1 = Standard_Real(theFaceVertices[n].m_Frac2)/Standard_Real( m_Data->GetZRGrid()->GetResolutionRatio() );
    tmpPnt[1] = tmpPnt[1] + tmpStep1*tmpFrac1;

    TxVector<Standard_Real> tmpXYZ;
    m_Data->GetZRDefine()->Convert_ZR_to_XYZ(tmpPnt, tmpXYZ);

	    
    BRepBuilderAPI_MakeVertex MV( gp_Pnt(tmpXYZ[0],tmpXYZ[1],tmpXYZ[2]) ); 
    if (MV.IsDone())  builder.Add(Comp,MV.Vertex());

  }


  ostringstream sstr;
  sstr<<"IntFaceVertices";
  sstr<<".brep";
  string s=sstr.str();
  
  BRepTools::Write(Comp,s.c_str());
}
