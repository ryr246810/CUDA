#include <ExprFuncBase.hxx>

ExprFuncBase::ExprFuncBase()
{
}

ExprFuncBase::
    ~ExprFuncBase()
{
}

void ExprFuncBase::
    init()
{
  m_vlist.AddDefaultValues();
  m_flist.AddDefaultFunctions();
}

void ExprFuncBase::
    setAttrib(const TxHierAttribSet &tas)
{
  tas.setToFirstOption();
  for (int i = 0; i < tas.getNumOptions(); i++)
  {
    std::pair<std::string, int> optionPair = tas.getCurrentOptionAndBump();
    string name = optionPair.first;
    double value = (double)optionPair.second;
    m_vlist.Add(name, value);
  }

  tas.setToFirstParam();
  for (int i = 0; i < tas.getNumParams(); i++)
  {
    std::pair<std::string, double> paramPair = tas.getCurrentParamAndBump();
    string name = paramPair.first;
    double value = paramPair.second;
    //cout<<"name = "<<name<<"  "<<"value = "<<value<<endl;
    m_vlist.Add(name, value);
  }

  m_expression.SetValueList(&m_vlist);
  m_expression.SetFunctionList(&m_flist);

  m_InputString.clear();

  string refName = "result";
  string kindName = "kind";

  if (tas.hasString(refName))
  {
    tas.setToFirstString();
    for (int i = 0; i < tas.getNumStrings(); i++)
    {
      std::pair<std::string, std::string> stringPair = tas.getCurrentStringAndBump();
      string name = stringPair.first;
      string value = stringPair.second;
      if ((name != refName) && (name != kindName))
      {
        string currExp = name + "=" + value + ";";
        m_InputString = m_InputString + currExp;
      }
    }
    string value = tas.getString(refName);
    string currExp = refName + "=" + value;
    m_InputString = m_InputString + currExp;
  }
  else if (tas.hasParam(refName))
  {
    tas.setToFirstParam();
    for (int i = 0; i < tas.getNumParams(); i++)
    {
      std::pair<std::string, double> stringPair = tas.getCurrentParamAndBump();
      string name = stringPair.first;
      double value = stringPair.second;
      if (name == refName)
      {
        string currExp;
        std::stringstream sstr;
        sstr << name
             << "="
             << value
             << ";";
        sstr >> currExp;

        m_InputString = m_InputString + currExp;
      }
    }
  }
  else
  {
    cout << "error---------------ExprFuncBase--------------no result argument is defined" << endl;
  }

  m_expression.Parse(m_InputString);
}

double
ExprFuncBase::
    evalute()
{
  double result = m_expression.Evaluate();
  return result;
}

double
ExprFuncBase::
    evalute(const double)
{
  return 0.0;
}

double
ExprFuncBase::
    evalute(const double, const double)
{
  return 0.0;
}

double
ExprFuncBase::
    evalute(const double, const double, const double)
{
  return 0.0;
}

double
ExprFuncBase::
    evalute(const double, const double, const double, const double)
{
  return 0.0;
}
