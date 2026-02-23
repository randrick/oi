use 5.12.0;
use strict;
use warnings;
use autodie;
use Data::Dumper;

my $hdr = <>;
my $record = <>;

# print $hdr . "\n";
# print $record . "\n";

my @h = map{trim($_)}split(/\|/, $hdr);
my @r = map{trim($_)}split(/\|/, $record);
    for my $i (0 .. $#h) {
        my $num_spaces = 32 - length($h[$i]);
        $h[$i] = sprintf("%$num_spaces"."s%s", " ", $h[$i]);
    }
# say Dumper(\@h);
# say Dumper(\@r);

die "Count mismatch" if scalar(@r) != scalar(@h);


my %data;
for my $i (0 .. $#h) {
    # say "l: " . length($h[$i]);
    # for (my $j=0; $j<32-length($h[$i]); $j++) {
    #     print " ";
    # }
    # say $h[$i] . ": " . $r[$i]."\n";
    printf("%-32s:  %s\n", $h[$i], $r[$i]);
}


sub trim {
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}

1;