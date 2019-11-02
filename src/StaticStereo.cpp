#include "utils/StaticStereo.hpp"

namespace cvo{
  static StaticStereo::TraceStatus StaticStereo::trace_stereo(const cv::Mat & left,
                                                              const cv::Mat & right,
                                                              const Mat33f & intrinsic,
                                                              const float baseline, // left->right < 0, right->left > 0
                                                              const pair<float, float> & input
                                                              pair<float, float> & result
                                                              ) const  {
    Vec3f bl;
    bl << baseline, 0, 0;
    float u_stereo = input.first;
    float v_stereo = input.second;
    float idepth_min = 0.01;

    // Kt: intrinsic * baseline extrinsict transform
    Vec3f Kt = intrinsic * bl;
    // T between stereo cameras
    Vec3f bl;
    
    // baseline * fx
    float bf = -K(0,0)*bl[0];

    Vec3f p_original = Vec3f(u_stereo,v_stereo, 1);
    Vec3f ptpMin = p_original +Kt * idepth_min;

    float uMin = ptpMin[0] / ptpMin[2];
    float vMin = ptpMin[1] / ptpMin[2];

    if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5))
    {
      lastTraceUV = Vec2f(-1,-1);
      lastTracePixelInterval=0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    float dist;
    float uMax;
    float vMax;
    Vec3f ptpMax;
    float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

    if(std::isfinite(idepth_max_stereo))
    {
      ptpMax = pr + Kt*idepth_max_stereo;
      uMax = ptpMax[0] / ptpMax[2];
      vMax = ptpMax[1] / ptpMax[2];


      if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
      {
        lastTraceUV = Vec2f(-1,-1);
        lastTracePixelInterval=0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
      }

      // ============== check their distance. everything below 2px is OK (-> skip). ===================
      dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
      dist = sqrtf(dist);
      if(dist < setting_trace_slackInterval)
      {
        //				lastTraceUV_Stereo = Vec2f(uMax+uMin, vMax+vMin)*0.5;
        //				lastTracePixelInterval_Stereo=dist;
        //				idepth_stereo = (u_stereo - 0.5*(uMax+uMin))/bf;
        //				return lastTraceStatus_Stereo = ImmaturePointStatus::IPS_SKIPPED;
        return lastTraceStatus = ImmaturePointStatus ::IPS_SKIPPED;

      }
      assert(dist>0);
    }
    else
    {
      dist = maxPixSearch;

      // project to arbitrary depth to get direction.
      ptpMax = pr + Kt*0.01;
      uMax = ptpMax[0] / ptpMax[2];
      vMax = ptpMax[1] / ptpMax[2];

      // direction.
      float dx = uMax-uMin;
      float dy = vMax-vMin;
      float d = 1.0f / sqrtf(dx*dx+dy*dy);

      // set to [setting_maxPixSearch].
      uMax = uMin + dist*dx*d;
      vMax = vMin + dist*dy*d;

      // may still be out!
      if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
      {
        lastTraceUV = Vec2f(-1,-1);
        lastTracePixelInterval=0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
      }
      assert(dist>0);
    }

    //		 set OOB if scale change too big.
    if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
    {
      lastTraceUV = Vec2f(-1, -1);
      lastTracePixelInterval = 0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
    float dx = setting_trace_stepsize*(uMax-uMin);
    float dy = setting_trace_stepsize*(vMax-vMin);

    float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
    float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
    float errorInPixel = 0.2f + 0.2f * (a+b) / a;

    if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max_stereo))
    {
      //			lastTraceUV_Stereo = Vec2f(uMax+uMin, vMax+vMin)*0.5;
      //			lastTracePixelInterval_Stereo=dist;
      //			idepth_stereo = (u_stereo - 0.5*(uMax+uMin))/bf;
      //			return lastTraceStatus_Stereo = ImmaturePointStatus::IPS_BADCONDITION;
      //            lastTraceUV = Vec2f(u, v);
      //            lastTracePixelInterval = dist;
      return lastTraceStatus = ImmaturePointStatus ::IPS_BADCONDITION;
    }

    if(errorInPixel >10) errorInPixel=10;

    // ============== do the discrete search ===================
    dx /= dist;
    dy /= dist;

    if(dist>maxPixSearch)
    {
      uMax = uMin + maxPixSearch*dx;
      vMax = vMin + maxPixSearch*dy;
      dist = maxPixSearch;
    }

    int numSteps = 1.9999f + dist / setting_trace_stepsize;
    Mat22f Rplane = KRKi.topLeftCorner<2,2>();

    float randShift = uMin*1000-floorf(uMin*1000);
    float ptx = uMin-randShift*dx;
    float pty = vMin-randShift*dy;


    Vec2f rotatetPattern[MAX_RES_PER_POINT];
    for(int idx=0;idx<patternNum;idx++)
      rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

    if(!std::isfinite(dx) || !std::isfinite(dy))
    {
      lastTraceUV = Vec2f(-1,-1);
      lastTracePixelInterval=0;
      return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    float errors[100];
    float bestU=0, bestV=0, bestEnergy=1e10;
    int bestIdx=-1;
    if(numSteps >= 100) numSteps = 99;

    for(int i=0;i<numSteps;i++)
    {
      float energy=0;
      for(int idx=0;idx<patternNum;idx++)
      {

        float hitColor = getInterpolatedElement31(frame->dI,
                                                  (float)(ptx+rotatetPattern[idx][0]),
                                                  (float)(pty+rotatetPattern[idx][1]),
                                                  wG[0]);

        if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
        float residual = hitColor - (float)(aff[0] * color[idx] + aff[1]);
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
        energy += hw *residual*residual*(2-hw);
      }

      errors[i] = energy;
      if(energy < bestEnergy)
      {
        bestU = ptx;
        bestV = pty;
        bestEnergy = energy;
        bestIdx = i;
      }

      ptx+=dx;
      pty+=dy;
    }

    // find best score outside a +-2px radius.
    float secondBest=1e10;
    for(int i=0;i<numSteps;i++)
    {
      if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
        secondBest = errors[i];
    }
    float newQuality = secondBest / bestEnergy;
    if(newQuality < quality || numSteps > 10) quality = newQuality;


    // ============== do GN optimization ===================
    float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
    if(setting_trace_GNIterations>0) bestEnergy = 1e5;
    int gnStepsGood=0, gnStepsBad=0;
    for(int it=0;it<setting_trace_GNIterations;it++)
    {
      float H = 1, b=0, energy=0;
      for(int idx=0;idx<patternNum;idx++)
      {
        Vec3f hitColor = getInterpolatedElement33(frame->dI,
                                                  (float)(bestU+rotatetPattern[idx][0]),
                                                  (float)(bestV+rotatetPattern[idx][1]),wG[0]);

        if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
        float residual = hitColor[0] - (aff[0] * color[idx] + aff[1]);
        float dResdDist = dx*hitColor[1] + dy*hitColor[2];
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

        H += hw*dResdDist*dResdDist;
        b += hw*residual*dResdDist;
        energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
      }


      if(energy > bestEnergy)
      {
        gnStepsBad++;

        // do a smaller step from old point.
        stepBack*=0.5;
        bestU = uBak + stepBack*dx;
        bestV = vBak + stepBack*dy;
      }
      else
      {
        gnStepsGood++;

        float step = -gnstepsize*b/H;
        if(step < -0.5) step = -0.5;
        else if(step > 0.5) step=0.5;

        if(!std::isfinite(step)) step=0;

        uBak=bestU;
        vBak=bestV;
        stepBack=step;

        bestU += step*dx;
        bestV += step*dy;
        bestEnergy = energy;

      }

      if(fabsf(stepBack) < setting_trace_GNThreshold) break;
    }

    if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
    {

      lastTracePixelInterval=0;
      lastTraceUV = Vec2f(-1,-1);
      if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
      else
        return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }

    // ============== set new interval ===================
    if(dx*dx>dy*dy)
    {
      idepth_min_stereo = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (Kt[0] - Kt[2]*(bestU-errorInPixel*dx));
      idepth_max_stereo = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (Kt[0] - Kt[2]*(bestU+errorInPixel*dx));
    }
    else
    {
      idepth_min_stereo = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (Kt[1] - Kt[2]*(bestV-errorInPixel*dy));
      idepth_max_stereo = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (Kt[1] - Kt[2]*(bestV+errorInPixel*dy));
    }
    if(idepth_min_stereo > idepth_max_stereo) std::swap<float>(idepth_min_stereo, idepth_max_stereo);



    if(!std::isfinite(idepth_min_stereo) || !std::isfinite(idepth_max_stereo) || (idepth_max_stereo<0) || (idepth_min_stereo < 0))
    {
      lastTracePixelInterval=0;
      lastTraceUV = Vec2f(-1,-1);
      return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }

    lastTracePixelInterval=2*errorInPixel;
    lastTraceUV = Vec2f(bestU, bestV);
    idepth_stereo = (u_stereo - bestU)/bf;
    //if (u > 790 && v > 300 && u < 880)
    //printf("(u,v): (%f, %f)->(%f, %f),  The idpeth_min is %f, the idepth_max is %f , idepth-stereo is %f, idepth_stero_min is %f, idepth_stereo_max is %f\n",
    //        u_stereo, v_stereo, lastTraceUV(0), lastTraceUV(1),  idepth_min, idepth_max, idepth_stereo, idepth_min_stereo, idepth_max_stereo);
    return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;

    
    
    
  }
  
}
