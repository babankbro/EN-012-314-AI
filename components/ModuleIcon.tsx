import {
  Boxes,
  BrainCircuit,
  Compass,
  Gamepad2,
  MessageSquareText,
  Network,
  ScanEye,
  Tags,
  TrendingUp,
  Workflow,
  type LucideIcon,
} from "lucide-react";

const ICONS: Record<string, LucideIcon> = {
  Boxes,
  BrainCircuit,
  Compass,
  Gamepad2,
  MessageSquareText,
  Network,
  ScanEye,
  Tags,
  TrendingUp,
  Workflow,
};

export default function ModuleIcon({ name, className }: { name: string; className?: string }) {
  const Icon = ICONS[name] ?? Compass;
  return <Icon className={className} />;
}
