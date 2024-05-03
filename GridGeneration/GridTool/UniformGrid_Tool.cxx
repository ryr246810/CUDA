#include <UniformGrid_Tool.hxx>

UniformGrid_Tool::
UniformGrid_Tool()
  : Grid_Tool()
{

}


UniformGrid_Tool::
~UniformGrid_Tool()
{

}


void 
UniformGrid_Tool::
SetAttrib(const TxHierAttribSet& tha)
{
  Grid_Tool::SetAttrib(tha);

  if(tha.hasOption("waveResolutionRatio")){
    m_WaveResolutionRatio = tha.getOption("waveResolutionRatio");
  }else{
    cout<<"error-----------UniformGrid_Tool::SetAttrib---------------1"<<endl;
  }
}


void
UniformGrid_Tool::
Build()
{
  const TxSlab<Standard_Real>& theBndBox = m_GridCtrl->GetBndBox();
  Standard_Real theMinWaveLength = m_GridCtrl->GetMinWaveLength();
  Standard_Integer rDir = m_GridCtrl->GetRDir();
  if(m_Dir!=rDir){
    m_Step = theMinWaveLength/(Standard_Real(m_WaveResolutionRatio));
    m_N = (Standard_Integer)(theBndBox.getLength(m_Dir)/m_Step) + 1;
    m_L = m_N * m_Step;
    //m_X0 = theBndBox.getLowerBound(m_Dir) - 0.5*(m_L - theBndBox.getLength(m_Dir));//tzh Modify 20210416
    m_X0 = theBndBox.getLowerBound(m_Dir) - 0.25*m_Step;
    for(Standard_Integer i=0; i<=m_N; i++){
      Standard_Real tmpX = m_X0 + m_Step*i;
      m_CoordinateVec.push_back(tmpX);
    }
  }else{  // r direction
    m_Step = theMinWaveLength/(Standard_Real(m_WaveResolutionRatio));
    m_N = (Standard_Integer)(theBndBox.getLength(m_Dir)/m_Step) + 1;
    m_Step = theBndBox.getLength(m_Dir)/m_N;
    m_L = m_N * m_Step;
    m_X0 = theBndBox.getLowerBound(m_Dir);
    for(Standard_Integer i=0; i<=m_N; i++){
      Standard_Real tmpX = m_X0 + m_Step*i;
      m_CoordinateVec.push_back(tmpX);
    }
  }
}
