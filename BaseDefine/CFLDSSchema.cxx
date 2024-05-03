/**
 * @file  CFLDSSchema.cpp
 *
 * @class CFLDSSchema
 *
 * @brief Describes how instant datasets and meshes are found in the
 * file compliant with the schema.
 */

#include <CFLDSSchema.hxx>


std::string CFLDSSchema::typeAtt ="type";
std::string CFLDSSchema::kindAtt = "kind";
std::string CFLDSSchema::meshAtt = "mesh";

std::string CFLDSSchema::nodeOffsetAtt = "nodeOffset";

std::string CFLDSSchema::centeringAtt ="centering";
std::string CFLDSSchema::indexOrderAtt = "indexOrder";

std::string CFLDSSchema::labelsAtt = "labels";
std::string CFLDSSchema::axisLabelsAtt = "axisLabels";


std::string CFLDSSchema::varKey = "variable";
std::string CFLDSSchema::varsKey = "vars";
std::string CFLDSSchema::meshKey = "mesh";

// New to CFLDSSchema 4.0
std::string CFLDSSchema::zonalCenteringKey = "zonal"; 
std::string CFLDSSchema::nodalCenteringKey = "nodal"; // Default
std::string CFLDSSchema::edgeCenteringKey = "edge";
std::string CFLDSSchema::faceCenteringKey = "face";


// Index ordering...
std::string CFLDSSchema::compMajorCKey = "compMajorC"; //currently not supported
std::string CFLDSSchema::compMinorCKey = "compMinorC"; //default ordering
std::string CFLDSSchema::compMajorFKey = "compMajorF"; //currently not supported
std::string CFLDSSchema::compMinorFKey = "compMinorF"; //supported


std::string CFLDSSchema::Uniform::key = "uniform";
std::string CFLDSSchema::Uniform::lowerBounds = "lowerBounds";
std::string CFLDSSchema::Uniform::startCell = "startCell";
std::string CFLDSSchema::Uniform::numCells = "numCells";
std::string CFLDSSchema::Uniform::upperBounds = "upperBounds";

//Time
std::string CFLDSSchema::timeKey = "time";
std::string CFLDSSchema::timeAtt = "Time";
std::string CFLDSSchema::cycleAtt = "Step";


//Run info
std::string CFLDSSchema::runInfoKey = "runInfo";
std::string CFLDSSchema::softwareAtt = "CEMPICSoftware";
