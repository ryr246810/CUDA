//--------------------------------------------------------------------
//
// File:	TModulatedTrapezePulse.cxx
//
// Purpose:	Implementation of a functor for a rising with the half-cosine in time.
//
//--------------------------------------------------------------------

#include <TModulatedTrapezePulse.hxx>
#include <PhysConsts.hxx>
#include <TxStreams.h>


TModulatedTrapezePulse::
TModulatedTrapezePulse()
{
  m_MS_Frequency = 0.;
  m_MS_StartTime = 0.;
  m_MS_EndTime = 0.;
  m_MS_Amp = 0.;
  m_MS_Phi = 0.;
  m_MS_OmegaRise = 0.;


  m_TP_TD = 0.;
  m_TP_TR = 0.;
  m_TP_TP = 0.;
  m_TP_TF = 0.;

  m_TP_Amp = 0.;
}


void 
TModulatedTrapezePulse::
setAttrib(const TxHierAttribSet& tas)
{
  // Set parameters of this object
  if(tas.hasParam("ms_startTime")) m_MS_StartTime = tas.getParam("startTime");

  Standard_Real theRiseTime = 0.0;
  if(tas.hasParam("ms_riseTime")) theRiseTime = tas.getParam("riseTime");
  m_MS_EndTime = theRiseTime + m_MS_StartTime;

  if(tas.hasParam("ms_amplitude")) m_MS_Amp = tas.getParam("amplitude");
  if(tas.hasParam("ms_phi")) m_MS_Phi = tas.getParam("phi");
  if(tas.hasParam("ms_frequency")) m_MS_Frequency = tas.getParam("frequency");


  // Get the amplitude inside the shape
  if(tas.hasParam("tp_amplitude")) m_TP_Amp = tas.getParam("amplitude");
  if(tas.hasParam("tp_delay")) m_TP_TD = tas.getParam("delay");
  if(tas.hasParam("tp_rise")) m_TP_TR = tas.getParam("rise");
  if(tas.hasParam("tp_pulse")) m_TP_TP = tas.getParam("pulse");
  if(tas.hasParam("tp_fall")) m_TP_TF = tas.getParam("fall");

  setup();
}


void 
TModulatedTrapezePulse::
setup()
{
  if(m_MS_EndTime == m_MS_StartTime) m_MS_OmegaRise = 0.;
  else m_MS_OmegaRise = mksConsts.piover2/(m_MS_EndTime - m_MS_StartTime);
}



Standard_Real 
TModulatedTrapezePulse::
operator()(Standard_Real t) const 
{
  // Get pulse amplitude
  Standard_Real ms_value = m_MS_Amp * sin(mksConsts.twopi * m_MS_Frequency * t + mksConsts.pi*m_MS_Phi);
  
  // Multiply times rising function.
  if(t <= m_MS_StartTime){
    ms_value=0.0;
  }else if(t >= m_MS_EndTime){
    // do nothing;
  }else{
    ms_value *= sin(m_MS_OmegaRise*(t-m_MS_StartTime))*sin(m_MS_OmegaRise*(t-m_MS_StartTime));
  }


  Standard_Real tp_value = 0.;
  Standard_Real t_tmp = m_TP_TD+m_TP_TR+m_TP_TP+m_TP_TF;

  if(t < m_TP_TD){
    tp_value = 0.;
  }else if( (t>=m_TP_TD) && (t<(m_TP_TD+m_TP_TR)) ){
    tp_value = m_TP_Amp*(t-m_TP_TD)/m_TP_TR;
  }else if( (t>=(m_TP_TD+m_TP_TR)) && (t<(m_TP_TD+m_TP_TR+m_TP_TP))) {
    tp_value =  m_TP_Amp;
  }else{
    tp_value = m_TP_Amp*(t_tmp-t)/m_TP_TF;
  }

  Standard_Real result = tp_value + ms_value;

  return result;
}

