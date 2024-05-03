
#include <ZRGrid_Ctrl.hxx>


/****************************************************************/
// Function : 
// Purpose  : 
/****************************************************************/
void ZRGrid_Ctrl::ComputeBndBoxAccrodingInputShapes()
{
  Standard_Real xmin,ymin,zmin,xmax,ymax,zmax;
  const TopTools_DataMapOfShapeInteger& theAllShapes =  m_ModelsCtrl->GetShapesWithType();

  if(theAllShapes.Extent ()!=0){
    BRep_Builder builder;	
    TopoDS_Compound Comp;	
    builder.MakeCompound(Comp);
    
    TopTools_DataMapIteratorOfDataMapOfShapeInteger Iter;
    for(Iter.Initialize(theAllShapes); Iter.More(); Iter.Next() ){
      const TopoDS_Shape & theShape = Iter.Key();
      builder.Add(Comp,theShape);
    }
    ComputeProperBndOfShape(Comp,xmin,ymin,zmin,xmax,ymax,zmax);
  }else{
    xmin = 0.0; xmax = 0.0;
    ymin = 0.0; ymax = 0.0;
    zmin = 0.0; zmax = 0.0;
  }

  TxVector<Standard_Real> ld_Pnt(xmin, ymin, zmin);
  TxVector<Standard_Real> ru_Pnt(xmax, ymax, zmax);

  m_Org[m_ZDir] = ld_Pnt[m_ZDir]; 

  ld_Pnt[m_RDir] = m_Org[m_RDir]; 
  ld_Pnt[m_WorkPlaneDir] = m_Org[m_WorkPlaneDir];
  ru_Pnt[m_WorkPlaneDir] = m_Org[m_WorkPlaneDir];

  m_BndBox.setBounds(ld_Pnt[0], ld_Pnt[1], ld_Pnt[2],  ru_Pnt[0], ru_Pnt[1], ru_Pnt[2]);




#ifdef MESH_CTRL_DBG
  cout<<"ZRGrid_Ctrl::ComputeBndBoxAccrodingInputShapes()---------Debug Inf-------->>"<<endl;
  m_BndBox.write(cout);
  cout<<"ZRGrid_Ctrl::ComputeBndBoxAccrodingInputShapes()---------Debug Inf--------<<"<<endl;
#endif
}

