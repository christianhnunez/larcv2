#ifndef __DIFFUSION3D_CXX__
#define __DIFFUSION3D_CXX__

#include "Diffusion3D.h"
#include "larcv/core/DataFormat/EventVoxel3D.h"
#include <algorithm>

namespace larcv {

  static Diffusion3DProcessFactory __global_Diffusion3DProcessFactory__;

  Diffusion3D::Diffusion3D(const std::string name)
  : ProcessBase(name)
  {}

  void Diffusion3D::configure_labels(const PSet& cfg)
  {
    _cluster3d_producer_v.clear();
    _output_producer_v.clear();
    _cluster3d_producer_v = cfg.get<std::vector<std::string> >("Cluster3DProducerList",_cluster3d_producer_v);
    _output_producer_v   = cfg.get<std::vector<std::string> >("OutputProducerList",_output_producer_v);
    _threshold = cfg.get<float>("Threshold",0);
    _division_threshold = cfg.get<float>("DivisionThreshold",0);
    if(_cluster3d_producer_v.empty()) {
      auto cluster3d_producer = cfg.get<std::string>("Cluster3DProducer","");
      auto output_producer   = cfg.get<std::string>("OutputProducer","");
      if(!cluster3d_producer.empty()) {
        _cluster3d_producer_v.push_back(cluster3d_producer);
        _output_producer_v.push_back(output_producer);
      }
    }

    if(_output_producer_v.empty()) {
      _output_producer_v.resize(_cluster3d_producer_v.size(),"");
    }
    else if(_output_producer_v.size() != _cluster3d_producer_v.size()) {
      LARCV_CRITICAL() << "Cluster3DProducer and OutputProducer must have the same array length!" << std::endl;
      throw larbys();
    }
  }

  void Diffusion3D::configure(const PSet& cfg)
  {
    configure_labels(cfg);
    auto sigma_v = cfg.get<std::vector<double> >("SigmaXYZ");  
    _numvox_v  = cfg.get<std::vector<size_t> >("NumVoxelsXYZ");
    _normalize = cfg.get<bool>("Normalize", true);
          
    // Load diffusion parameters from config.
    //auto Dlong = cfg.get<double>("DiffLong"); // 7.2 cm^2/s  for E = 500V/cm
    //auto Dtran = cfg.get<double>("DiffTran"); // 12  cm^2/s  for E = 500V/cm
    //auto vdrift = cfg.get<double>("DriftVelocity"); // 0.1113 cm/us (SimIonization)
      
    if (sigma_v.size() != 3) {
      LARCV_CRITICAL() << "SigmaXYZ parameter must be length 3 floating point vector!" << std::endl;
      throw larbys();
    }
    if (_numvox_v.size() != 3) {
      LARCV_CRITICAL() << "NumVoxelsXYZ parameter must be length 3 unsigned integer vector!" << std::endl;
      throw larbys();
    }

    // Precalculate sigma ** 2
    std::transform(sigma_v.cbegin(), sigma_v.cend(), std::back_inserter(_sigma2_v),
                   [](double v) { return v * v; });

    // Create smearing matrix
    _scale_vvv.resize(_numvox_v[0] + 1);
    for (auto& scale_vv : _scale_vvv) {
      scale_vv.resize(_numvox_v[1] + 1);
      for (auto& scale_v : scale_vv) {
        scale_v.resize(_numvox_v[2] + 1, 0.);
      }
    } 
  }

  void Diffusion3D::initialize()
  {}

  bool Diffusion3D::process(IOManager& mgr)
  {

    for (size_t producer_index = 0; producer_index < _cluster3d_producer_v.size(); ++producer_index) {

      auto const& cluster3d_producer = _cluster3d_producer_v[producer_index];
      auto const& output_producer   = _output_producer_v[producer_index];

      auto const& ev_cluster3d = mgr.get_data<larcv::EventClusterVoxel3D>(cluster3d_producer);
      auto ev_output = (larcv::EventClusterVoxel3D*)(mgr.get_data("cluster3d",output_producer));

      if(ev_output->meta().valid()) {
        static bool one_time_warning=true;
        if(_output_producer_v[producer_index].empty()) {
          LARCV_CRITICAL() << "Over-writing existing EventSparseCluster3D data for label "
          << output_producer << std::endl;
          throw larbys();
        }
        else if(one_time_warning) {
          LARCV_WARNING() << "Output EventSparseCluster3D producer " << output_producer
          << " already holding valid data will be over-written!" << std::endl;
          one_time_warning = false;
        }
      }
        
//       // ev_cluster3d is EventClusterVoxel3D, see Voxel3DMeta.h  for contents of meta.
       auto const& meta = ev_cluster3d.meta();
    
//       // For this ev_cluster3d, get varL and varT
//       auto xpos = meta.pos_x() // x-axis is drift direction
//       auto t = xpos/_vdrift    // drift time in microseconds
//       auto varL = 2 * _d_l * t // longitudinal spatial diffusion variance
//       auto varT = 2 * _d_t * t // transverse spatial diffusion variance
        
//       for (size_t xshift = 0; xshift <= _numvox_v[0]; ++xshift) {
//         for (size_t yshift = 0; yshift <= _numvox_v[1]; ++yshift) {
//           for (size_t zshift = 0; zshift <= _numvox_v[2]; ++zshift) {

//             double val = exp( - pow(xshift * meta.size_voxel_x(), 2) / (2. * varL)
//               - pow(yshift * meta.size_voxel_y(), 2) / (2. * varT)
//               - pow(zshift * meta.size_voxel_z(), 2) / (2. * varT) );
//             _scale_vvv[xshift][yshift][zshift] = val;
//           }
//         }
//       }

//       double scale_sum = 1.;
//       if(_normalize) {
//        scale_sum = 0.;
//        int x_ctr = -((int)_numvox_v[0]);
//        while(x_ctr <= (int)(_numvox_v[0])) {
//          int y_ctr = -((int)_numvox_v[1]);
//          while(y_ctr <= (int)(_numvox_v[1])) {
//            int z_ctr = -((int)_numvox_v[2]);
//            while(z_ctr <= (int)(_numvox_v[2])) {
//              scale_sum += _scale_vvv[std::abs(x_ctr)][std::abs(y_ctr)][std::abs(z_ctr)];
//              ++z_ctr;
//            }
//            ++y_ctr;
//          }
//          ++x_ctr;
//        }
//      }

     // LARCV_INFO() << "scale_sum: " << scale_sum << std::endl;

     ev_output->meta(meta);
     std::vector<larcv::VoxelSet> vsa_output;
     for(size_t cluster_index=0; cluster_index<ev_cluster3d.as_vector().size(); ++cluster_index) {

       auto const& cluster = ev_cluster3d.as_vector()[cluster_index];
       larcv::VoxelSet res_data;
       for (auto const& vox : cluster.as_vector()) {
         LARCV_DEBUG() << "Re-mapping vox ID " << vox.id() << " charge " << vox.value() << std::endl;
         float sum_charge = 0.;
         auto const pos = meta.position(vox.id());
         double xpos = pos.x - _numvox_v[0] * meta.size_voxel_x();
         double xmax = pos.x + (_numvox_v[0] + 0.5) * meta.size_voxel_x();
         int x_ctr = 0;
           
         // ======== BEGIN: Diffusion Matrix ========
         // For each voxel, create the smearing matrix:
         // For this voxel, get varL and varT
         
         // FIX [hardcode]
         double DriftVelocity = 0.1113; // cm/us ... 0.16
         double DiffLong = 0.0000072;   // cm^2/us  for E = 500V/cm
         double DiffTran = 0.000012;  // cm^2/us  for E = 500V/cm
           
         auto true_x = pos.x; // x-axis is drift direction
           
         // If default, subtract from bounding box
         auto x_rel_anode = true_x - meta.x_org(vox.id());
             
         auto time = std::abs(x_rel_anode)/DriftVelocity;    // drift time in microseconds
         auto varL = 2. * DiffLong * time; // longitudinal spatial diffusion variance
         auto varT = 2. * DiffTran * time; // transverse spatial diffusion variance
           
         //LARCV_INFO() << "true_x = " << true_x << "; varL = " << varL << "; varT = " << varT << "; org_x = " << meta.x_org(vox.id()) << "; x_rel_anode = " << x_rel_anode << std::endl;
         LARCV_INFO() << varL; 
        
         // Constructing matrix
         for (size_t xshift = 0; xshift <= _numvox_v[0]; ++xshift) {
            for (size_t yshift = 0; yshift <= _numvox_v[1]; ++yshift) {
              for (size_t zshift = 0; zshift <= _numvox_v[2]; ++zshift) {
                  
                double val = exp( - pow(xshift * meta.size_voxel_x(), 2) / (2. * varL)
                  - pow(yshift * meta.size_voxel_y(), 2) / (2. * varT)
                  - pow(zshift * meta.size_voxel_z(), 2) / (2. * varT) );
                _scale_vvv[xshift][yshift][zshift] = val;
              
              }
            }
         }
         
         // Normalization (for charge)
         double scale_sum = 1.0;
         if(_normalize) {
           scale_sum = 0.;
           int x_ctr = -((int)_numvox_v[0]);
           while(x_ctr <= (int)(_numvox_v[0])) {
             int y_ctr = -((int)_numvox_v[1]);
             while(y_ctr <= (int)(_numvox_v[1])) {
               int z_ctr = -((int)_numvox_v[2]);
               while(z_ctr <= (int)(_numvox_v[2])) {
                 scale_sum += _scale_vvv[std::abs(x_ctr)][std::abs(y_ctr)][std::abs(z_ctr)];
                 ++z_ctr;
               }
               ++y_ctr;
             }
             ++x_ctr;
           }
         }
         // ======== END: Diffusion Matrix ========
          
         while (xpos < xmax) {
           double ypos = pos.y - _numvox_v[1] * meta.size_voxel_y();
           double ymax = pos.y + (_numvox_v[1] + 0.5) * meta.size_voxel_y();
           int y_ctr = 0;
           while (ypos < ymax) {
             double zpos = pos.z - _numvox_v[2] * meta.size_voxel_z();
             double zmax = pos.z + (_numvox_v[2] + 0.5) * meta.size_voxel_z();
             int z_ctr = 0;
             while (zpos < zmax) {
              
              auto const id = meta.id(xpos, ypos, zpos);
              if (id != kINVALID_VOXELID) {
                
                int xindex = std::abs(((int)(_numvox_v[0])) - x_ctr);
                int yindex = std::abs(((int)(_numvox_v[1])) - y_ctr);
                int zindex = std::abs(((int)(_numvox_v[2])) - z_ctr);
                
                float scale_factor = _scale_vvv[xindex][yindex][zindex];
                float charge = vox.value() * scale_factor / scale_sum;

                if(charge > _division_threshold) {
                  LARCV_DEBUG() << "... to ID " << id << " charge " << charge << std::endl;
                  res_data.emplace(id, charge, true);
                  sum_charge += charge;
                }
              }
              zpos += meta.size_voxel_z();
              ++z_ctr;
            }
            ypos += meta.size_voxel_y();
            ++y_ctr;
          }
          xpos += meta.size_voxel_x();
          ++x_ctr;
        }
        LARCV_DEBUG() << "Re-mapped sum charge " << sum_charge << std::endl;
      }
	// Only keep voxels above threshold
      VoxelSet res_data_threshold;
      for(auto const& vox : res_data.as_vector()) {
       if(vox.value() <= _threshold) continue;
       res_data_threshold.emplace(vox.id(), vox.value(), true);
     }

     LARCV_INFO() << "Before: vox count = " << cluster.as_vector().size() << " charge = " << cluster.sum()
     << " ... "
     << res_data.sum() << " ... " 
     << "After: vox count = " << res_data_threshold.as_vector().size() << " charge = " << res_data_threshold.sum() << std::endl;
     
     vsa_output.emplace_back(std::move(res_data));
   }
   ((VoxelSetArray*)(ev_output))->emplace(std::move(vsa_output));
 }
 LARCV_INFO() << "DONE" << std::endl;
 return true;
 
}

void Diffusion3D::finalize()
{}

}
#endif
