#include <TTrapezePulse.hxx>
#include <TxStreams.h>

TTrapezePulse::
TTrapezePulse()
{
  m_TD = 0.;
  m_TR = 0.;
  m_TP = 0.;
  m_TF = 0.;

  m_Amp = 0.;
}



void 
TTrapezePulse::
setAttrib(const TxHierAttribSet& tas)
{
  // Get the amplitude inside the shape
  if(tas.hasParam("amplitude")) m_Amp = tas.getParam("amplitude");
  if(tas.hasParam("delay")) m_TD = tas.getParam("delay");
  if(tas.hasParam("rise")) m_TR = tas.getParam("rise");
  if(tas.hasParam("pulse")) m_TP = tas.getParam("pulse");
  if(tas.hasParam("fall")) m_TF = tas.getParam("fall");
}




Standard_Real 
TTrapezePulse::
operator()( Standard_Real t) const 
{
  Standard_Real t_tmp = 0.;

  if(t < m_TD)    return 0.;
  if(t < (t_tmp = m_TD+m_TR)) return m_Amp*(t-m_TD)/m_TR;
  if(t < (t_tmp += m_TP))     return m_Amp;
  if(t < (t_tmp += m_TF))     return m_Amp*(t_tmp-t)/m_TF;

  return 0.;
}
